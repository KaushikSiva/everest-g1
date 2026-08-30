import pytest

from everest_g1.bright_data import BrightDataConfigurationError, BrightDataSettings


def test_bright_data_requires_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BRIGHT_DATA_API_TOKEN", raising=False)
    with pytest.raises(BrightDataConfigurationError):
        BrightDataSettings.from_env()


def test_bright_data_is_allowlisted_and_repr_redacts_token() -> None:
    settings = BrightDataSettings("super-secret")
    url = settings.remote_url()

    assert "tools=search_engine%2Cscrape_as_markdown" in url
    assert "super-secret" in url
    assert "super-secret" not in repr(settings)
