"""Local MCP server exposing read tools and enqueue-only control tools."""

from __future__ import annotations

import argparse
from contextlib import suppress
from pathlib import Path
from typing import Any, Literal

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from summit_sentinel.bridge import SQLiteBridge

READ_ONLY_TOOL = ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)
ENQUEUE_ONLY_TOOL = ToolAnnotations(
    readOnlyHint=False,
    destructiveHint=False,
    idempotentHint=False,
    openWorldHint=False,
)
RESET_ENQUEUE_TOOL = ToolAnnotations(
    readOnlyHint=False,
    destructiveHint=True,
    idempotentHint=False,
    openWorldHint=False,
)
ControlMode = Literal["hold", "supervisory"]


class StrictArgumentMCPServer(MCPServer):
    """Reject fields outside every advertised top-level tool schema.

    MCP SDK 2.1.1 exposes ``list_tools`` and ``call_tool`` as async server
    methods, and its request handler dispatches through ``call_tool``. The SDK's
    generated function argument model otherwise uses Pydantic's default
    ``extra='ignore'``. Keeping this check at the server method makes the schema
    truthful and covers both direct and protocol calls without private manager
    mutation. Source:
    https://github.com/modelcontextprotocol/python-sdk/blob/v2.1.1/src/mcp/server/mcpserver/server.py#L497-L533
    """

    async def list_tools(self):
        tools = await super().list_tools()
        for tool in tools:
            tool.input_schema["additionalProperties"] = False
        return tools

    async def call_tool(self, name: str, arguments: dict[str, Any], context=None):
        schemas = {tool.name: tool.input_schema for tool in await self.list_tools()}
        schema = schemas.get(name)
        if schema is not None:
            allowed = set(schema.get("properties", {}))
            extras = set(arguments) - allowed
            if extras:
                fields = ", ".join(sorted(extras))
                raise ToolError(f"unexpected top-level tool fields: {fields}")
        return await super().call_tool(name, arguments, context)


def build_mcp_server(database_path: Path) -> MCPServer:
    """Build an MCP SDK v2 server around a SQLite queue, never an environment.

    API provenance (official SDK 2.1.1): ``FastMCP`` was renamed to
    ``mcp.server.MCPServer`` while ``@mcp.tool()`` remains the high-level tool
    decorator. See https://github.com/modelcontextprotocol/python-sdk/blob/v2.1.1/docs/whats-new.md
    """

    bridge = SQLiteBridge(database_path)
    mcp = StrictArgumentMCPServer(
        "Summit Sentinel Agent Bridge",
        version="0.2.0",
        instructions=(
            "Connector-ready operator surface. Reads return bounded local state. "
            "Writes require an external approval reference and only enqueue validated "
            "commands; they never access MuJoCo or clear local/fault stop authority."
        ),
    )

    @mcp.tool(annotations=READ_ONLY_TOOL)
    def get_sim_state() -> dict[str, object]:
        """Read bridge/run health and the latest bounded simulator frame."""

        return bridge.status()

    @mcp.tool(annotations=READ_ONLY_TOOL)
    def get_joystick_state() -> dict[str, object]:
        """Read bounded calibrated joystick state, including disconnected state."""

        return bridge.joystick_state()

    @mcp.tool(annotations=READ_ONLY_TOOL)
    def get_run_telemetry(limit: int = 100, run_id: str | None = None) -> dict[str, object]:
        """Read at most 1000 structured frames and mission metrics for one run."""

        return bridge.run_telemetry(limit, run_id=run_id)

    @mcp.tool(annotations=READ_ONLY_TOOL)
    def export_replay(limit: int = 500, run_id: str | None = None) -> dict[str, object]:
        """Return a bounded structured replay; never writes an arbitrary path."""

        replay = bridge.run_telemetry(limit, run_id=run_id)
        return {
            "format": "summit-sentinel-replay/v1",
            "storage": {
                "kind": "sqlite",
                "database_name": bridge.path.name,
                "logical_path": (
                    f"telemetry://runs/{replay['run_id']}"
                    if replay["run_id"] is not None
                    else "telemetry://runs/empty"
                ),
                "filesystem_write": False,
            },
            **replay,
        }

    @mcp.tool(annotations=RESET_ENQUEUE_TOOL)
    def apply_scenario_conditions(
        friction: float,
        wind_mps: float,
        visibility_m: float,
        snow_depth_m: float,
        approval_ref: str,
    ) -> dict[str, object]:
        """Enqueue clamped environmental conditions after operator approval.

        ``approval_ref`` is an opaque external audit reference, not authentication.
        The MCP process never accesses MuJoCo.
        """

        return bridge.enqueue_command(
            "scenario_conditions",
            {
                "friction": friction,
                "wind_mps": wind_mps,
                "visibility_m": visibility_m,
                "snow_depth_m": snow_depth_m,
                "approval_ref": approval_ref,
            },
        )

    @mcp.tool(annotations=RESET_ENQUEUE_TOOL)
    def set_control_mode(mode: ControlMode, approval_ref: str) -> dict[str, object]:
        """Enqueue hold/supervisory mode after operator approval.

        This cannot clear a local/fault stop or satisfy local reset acknowledgement.
        """

        return bridge.enqueue_command("control_mode", {"mode": mode, "approval_ref": approval_ref})

    @mcp.tool(annotations=RESET_ENQUEUE_TOOL)
    def reset_simulation(approval_ref: str) -> dict[str, object]:
        """Enqueue destructive state reset after explicit operator approval.

        Reset never clears or acknowledges a local/fault stop.
        """

        return bridge.enqueue_command("reset", {"approval_ref": approval_ref})

    @mcp.tool(annotations=ENQUEUE_ONLY_TOOL)
    def request_remote_stop(approval_ref: str) -> dict[str, object]:
        """Enqueue a best-effort supervisory stop request.

        Delivery is asynchronous and non-deterministic. This never claims to be
        the simulator's local emergency stop and cannot acknowledge that stop.
        """

        return bridge.enqueue_command("remote_stop", {"approval_ref": approval_ref})

    # Official MCP SDK 2.1.1: custom routes are async Starlette
    # Request -> Response handlers and are intentionally unauthenticated.
    # https://github.com/modelcontextprotocol/python-sdk/blob/v2.1.1/docs/run/asgi.md#custom-routes
    @mcp.custom_route("/health", methods=["GET"])
    async def health(request: Request) -> Response:
        del request
        return JSONResponse(bridge.health())

    return mcp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-db", type=Path, default=Path("summit-sentinel.db"))
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--mcp-path", default="/mcp")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be between 1 and 65535")
    if not args.mcp_path.startswith("/") or "//" in args.mcp_path:
        raise SystemExit("--mcp-path must be an absolute URL path")
    mcp = build_mcp_server(args.bridge_db)
    # This exact overload is documented by SDK 2.1.1. Binding is deliberately
    # fixed to loopback. Streamable HTTP supersedes the legacy SSE transport.
    # https://github.com/modelcontextprotocol/python-sdk/blob/v2.1.1/docs/run/index.md
    with suppress(KeyboardInterrupt):
        mcp.run(
            transport="streamable-http",
            host="127.0.0.1",
            port=args.port,
            streamable_http_path=args.mcp_path,
            json_response=True,
            stateless_http=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
