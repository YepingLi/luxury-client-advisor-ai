"""Configuration routing tests use fictional keys and never open a connection."""

import pytest

from ca_tool.settings import PROJECT_ROOT, Settings


@pytest.mark.parametrize("provider,expected", [
    ("openrouter", "https://openrouter.ai/api/v1"),
    ("tokenrouter", "https://api.tokenrouter.com/v1"),
    (" TOKENROUTER ", "https://api.tokenrouter.com/v1"),
])
def test_provider_base_url_defaults(provider, expected):
    assert Settings(provider=provider).base_url == expected


def test_explicit_base_url_overrides_default_and_normalizes_trailing_slash():
    settings = Settings(provider="tokenrouter", base_url="https://gateway.example/v1/")
    assert settings.base_url == "https://gateway.example/v1"


def test_generic_provider_requires_explicit_base_url():
    with pytest.raises(ValueError, match="LLM_BASE_URL is required"):
        Settings(provider="openai_compatible")
    assert Settings(provider="openai_compatible", base_url="http://localhost:8000/v1").base_url == "http://localhost:8000/v1"
    assert Settings(provider="openai_compatible", base_url="http://[::1]:8000/v1").base_url == "http://[::1]:8000/v1"


@pytest.mark.parametrize("value", [
    "ftp://gateway.example/v1",
    "gateway.example/v1",
    "https:///v1",
    "https://user:password@gateway.example/v1",
    "https://user@gateway.example/v1",
    "https://gateway.example/v1?api_key=fictional",
    "https://gateway.example/v1?",
    "https://gateway.example/v1#fragment",
    "https://gateway.example/v1#",
    "https://gateway.example/v1/chat/completions",
    "https://gateway.example/v1/chat/completions/",
    "https://gateway.example/v1/%63hat/completions",
    "https://gateway.example:invalid/v1",
    "https://gateway.example:65536/v1",
    "https://gateway.example:0/v1",
    "https://gate way.example/v1",
    "https://gateway.example/v1\n",
    "https://gateway.example\\other/v1",
    "https://gateway..example/v1",
    "https://-gateway.example/v1",
    "https://[invalid]/v1",
])
def test_invalid_or_completion_endpoint_base_urls_are_rejected(value):
    with pytest.raises(ValueError, match="LLM_BASE_URL"):
        Settings(provider="tokenrouter", base_url=value)


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    for name in (
        "LLM_PROVIDER", "LLM_BASE_URL", "LLM_API_KEY", "TOKENROUTER_API_KEY", "OPENROUTER_API_KEY",
        "LLM_MODEL", "LLM_MAX_ATTEMPTS", "LLM_TIMEOUT_SECONDS", "LLM_TOTAL_TIMEOUT_SECONDS",
        "MAX_INPUT_BYTES", "LLM_MAX_TOKENS", "LLM_TEMPERATURE", "LLM_MAX_TOTAL_CALLS", "RUNS_DIR", "BRAND_PATH",
        "GOOGLE_CLOUD_API_KEY", "GOOGLE_API_KEY", "GOOGLE_CLOUD_PROJECT", "GOOGLE_CLOUD_LOCATION", "VERTEX_AUTH_MODE",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GEMINI_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / "settings.env"
    path.write_text("", encoding="utf-8")
    return path


def test_openrouter_keeps_legacy_key_alias_priority(env_file):
    env_file.write_text("OPENROUTER_API_KEY=fictional-openrouter\nLLM_API_KEY=fictional-generic\n", encoding="utf-8")
    settings = Settings.from_env(env_file)
    assert settings.api_key == "fictional-openrouter"
    assert settings.base_url == "https://openrouter.ai/api/v1"


def test_tokenrouter_prefers_explicit_generic_key_then_matching_alias(env_file):
    env_file.write_text("LLM_PROVIDER=tokenrouter\nTOKENROUTER_API_KEY=fictional-tokenrouter\nLLM_API_KEY=fictional-generic\nOPENROUTER_API_KEY=fictional-unrelated\n", encoding="utf-8")
    assert Settings.from_env(env_file).api_key == "fictional-generic"
    env_file.write_text("LLM_PROVIDER=tokenrouter\nTOKENROUTER_API_KEY=fictional-tokenrouter\n", encoding="utf-8")
    assert Settings.from_env(env_file).api_key == "fictional-tokenrouter"


@pytest.mark.parametrize("provider,unrelated_variable", [
    ("tokenrouter", "OPENROUTER_API_KEY"),
    ("openrouter", "TOKENROUTER_API_KEY"),
    ("openai_compatible", "OPENROUTER_API_KEY"),
    ("openai_compatible", "TOKENROUTER_API_KEY"),
])
def test_provider_does_not_implicitly_send_another_gateways_credential(env_file, provider, unrelated_variable):
    env_file.write_text(f"LLM_PROVIDER={provider}\nLLM_BASE_URL=https://gateway.example/v1\n{unrelated_variable}=fictional-unrelated\n", encoding="utf-8")
    assert Settings.from_env(env_file).api_key == ""


def test_process_environment_overrides_dotenv_values(env_file, monkeypatch):
    env_file.write_text("LLM_PROVIDER=tokenrouter\nLLM_BASE_URL=https://file.example/v1\nLLM_API_KEY=fictional-file\n", encoding="utf-8")
    monkeypatch.setenv("LLM_BASE_URL", "https://process.example/v1")
    monkeypatch.setenv("LLM_API_KEY", "fictional-process")
    settings = Settings.from_env(env_file)
    assert settings.base_url == "https://process.example/v1"
    assert settings.api_key == "fictional-process"
    assert "fictional-process" not in repr(settings)


def test_unknown_provider_is_explicit_configuration_error():
    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        Settings(provider="unsupported")


def test_vertex_defaults_to_current_low_cost_model_and_global_google_host():
    settings = Settings(provider="vertex")
    assert settings.model == "gemini-3.1-flash-lite"
    assert settings.base_url == "https://aiplatform.googleapis.com/v1"
    assert settings.vertex_auth_mode == "adc"
    assert settings.vertex_project == ""  # Offline input preflight needs no credentials.


def test_vertex_physical_region_has_matching_google_host():
    settings = Settings(provider="vertex", vertex_location="us-central1")
    assert settings.base_url == "https://us-central1-aiplatform.googleapis.com/v1"


@pytest.mark.parametrize("location", ["us", "eu", "../global", "global/other", "global?key=other", "https://other.example", "us-central1.evil.example", "us-central1\n"])
def test_vertex_location_rejects_unimplemented_multiregions_and_endpoint_injection(location):
    with pytest.raises(ValueError, match="GOOGLE_CLOUD_LOCATION"):
        Settings(provider="vertex", vertex_location=location)


@pytest.mark.parametrize("url", [
    "https://unrelated.example/v1",
    "http://aiplatform.googleapis.com/v1",
    "https://aiplatform.googleapis.com.evil.example/v1",
    "https://us-central1-aiplatform.googleapis.com/v1",
    "https://aiplatform.googleapis.com/v1/other",
    "https://aiplatform.googleapis.com:443/v1",
])
def test_vertex_base_url_cannot_redirect_google_credentials_to_a_different_route(url):
    with pytest.raises(ValueError, match="Vertex LLM_BASE_URL"):
        Settings(provider="vertex", base_url=url)


def test_vertex_accepts_only_matching_explicit_base_url():
    assert Settings(provider="vertex", base_url="https://aiplatform.googleapis.com/v1/").base_url == "https://aiplatform.googleapis.com/v1"
    settings = Settings(provider="vertex", vertex_location="us-central1", base_url="https://us-central1-aiplatform.googleapis.com/v1")
    assert settings.base_url == "https://us-central1-aiplatform.googleapis.com/v1"


@pytest.mark.parametrize("field,value", [
    ("vertex_project", "project/locations/other"),
    ("vertex_project", "project?key=value"),
    ("vertex_project", "project\n"),
    ("model", "publishers/google/models/gemini-3.1-flash-lite"),
    ("model", "gemini-3.1-flash-lite:generateContent"),
    ("model", "gemini-3.1-flash-lite?key=value"),
    ("model", "../gemini-3.1-flash-lite"),
    ("model", "gemini-3.1-flash-lite\n"),
    ("vertex_auth_mode", "unsupported"),
])
def test_vertex_identifiers_cannot_inject_endpoint_segments(field, value):
    with pytest.raises(ValueError):
        Settings(provider="vertex", **{field: value})


@pytest.mark.parametrize("credential_name", ["GOOGLE_CLOUD_API_KEY", "GOOGLE_API_KEY"])
def test_vertex_env_google_key_selects_api_key_auth_and_current_model(env_file, credential_name):
    env_file.write_text(f"LLM_PROVIDER=vertex\n{credential_name}=fictional-google-key\n", encoding="utf-8")
    settings = Settings.from_env(env_file)
    assert settings.api_key == "fictional-google-key"
    assert settings.vertex_auth_mode == "api_key"
    assert settings.model == "gemini-3.1-flash-lite"


def test_vertex_prefers_google_cloud_alias_and_never_reuses_old_gateway_keys(env_file):
    env_file.write_text("LLM_PROVIDER=vertex\nGOOGLE_CLOUD_API_KEY=fictional-cloud-key\nGOOGLE_API_KEY=fictional-google-alias\nLLM_API_KEY=fictional-old-key\nTOKENROUTER_API_KEY=fictional-tokenrouter\nOPENROUTER_API_KEY=fictional-openrouter\n", encoding="utf-8")
    assert Settings.from_env(env_file).api_key == "fictional-cloud-key"
    env_file.write_text("LLM_PROVIDER=vertex\nLLM_API_KEY=fictional-old-key\nTOKENROUTER_API_KEY=fictional-tokenrouter\nOPENROUTER_API_KEY=fictional-openrouter\n", encoding="utf-8")
    settings = Settings.from_env(env_file)
    assert settings.api_key == ""
    assert settings.vertex_auth_mode == "adc"


def test_vertex_explicit_adc_overrides_key_presence_without_loading_credentials(env_file):
    env_file.write_text("LLM_PROVIDER=vertex\nGOOGLE_CLOUD_API_KEY=fictional-google-key\nGOOGLE_CLOUD_PROJECT=fictional-project\nGOOGLE_CLOUD_LOCATION=us-central1\nVERTEX_AUTH_MODE=adc\nLLM_MODEL=gemini-3.5-flash-lite\n", encoding="utf-8")
    settings = Settings.from_env(env_file)
    assert settings.vertex_auth_mode == "adc"
    assert settings.vertex_project == "fictional-project"
    assert settings.vertex_location == "us-central1"
    assert settings.model == "gemini-3.5-flash-lite"


def test_vertex_route_identity_hashes_project_and_excludes_credentials():
    import hashlib
    settings = Settings(provider="vertex", api_key="fictional-google-secret", vertex_project="fictional-project")
    assert settings.route_identity == {
        "provider": "vertex", "base_url": "https://aiplatform.googleapis.com/v1",
        "vertex_location": "global", "vertex_auth_mode": "adc",
        "vertex_project_sha256": hashlib.sha256(b"fictional-project").hexdigest(),
    }
    assert "fictional-project" not in str(settings.route_identity)
    assert "fictional-google-secret" not in repr(settings)
    assert "fictional-project" not in repr(settings)


@pytest.mark.parametrize("provider", ["openrouter", "tokenrouter", "openai_compatible"])
def test_legacy_route_identity_keeps_exact_two_field_contract(provider):
    settings = Settings(provider=provider, base_url="https://gateway.example/v1")
    assert settings.route_identity == {"provider": provider, "base_url": "https://gateway.example/v1"}


def test_google_keys_do_not_populate_legacy_gateway_credentials(env_file):
    env_file.write_text("GOOGLE_CLOUD_API_KEY=fictional-google-key\nGOOGLE_API_KEY=fictional-google-alias\n", encoding="utf-8")
    assert Settings.from_env(env_file).api_key == ""


def test_vertex_relative_adc_file_in_dotenv_resolves_against_project_root(env_file):
    env_file.write_text("LLM_PROVIDER=vertex\nVERTEX_AUTH_MODE=adc\nGOOGLE_APPLICATION_CREDENTIALS=private/fictional-service-account.json\n", encoding="utf-8")
    settings = Settings.from_env(env_file)
    assert settings.vertex_credentials_path == (PROJECT_ROOT / "private/fictional-service-account.json").resolve()
    assert "fictional-service-account" not in repr(settings)
    assert "private/fictional-service-account" not in str(settings.route_identity)
    assert "vertex_credentials_path" not in settings.route_identity


def test_vertex_process_adc_filepath_overrides_dotenv_without_opening_file(env_file, tmp_path, monkeypatch):
    private_path = tmp_path / "fictional-private-credentials.json"
    assert not private_path.exists()
    env_file.write_text("LLM_PROVIDER=vertex\nGOOGLE_APPLICATION_CREDENTIALS=other/fictional-old.json\n", encoding="utf-8")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(private_path))
    settings = Settings.from_env(env_file)
    assert settings.vertex_credentials_path == private_path.resolve()
    assert str(private_path) not in repr(settings)
    assert str(private_path) not in str(settings.route_identity)
    assert not private_path.exists()


def test_gemini_developer_api_has_canonical_endpoint_and_default_model():
    settings = Settings(provider="gemini")
    assert settings.base_url == "https://generativelanguage.googleapis.com/v1beta"
    assert settings.model == "gemini-3.1-flash-lite"
    assert settings.route_identity == {
        "provider": "gemini", "base_url": "https://generativelanguage.googleapis.com/v1beta",
    }


@pytest.mark.parametrize("url", [
    "https://unrelated.example/v1beta",
    "https://generativelanguage.googleapis.com.evil.example/v1beta",
    "https://aiplatform.googleapis.com/v1",
    "http://generativelanguage.googleapis.com/v1beta",
    "https://generativelanguage.googleapis.com:443/v1beta",
    "https://generativelanguage.googleapis.com/v1",
    "https://generativelanguage.googleapis.com/v1beta/models",
    "https://user:fictional-password@generativelanguage.googleapis.com/v1beta",
    "https://generativelanguage.googleapis.com/v1beta?key=fictional-key",
    "https://generativelanguage.googleapis.com/v1beta#fragment",
])
def test_gemini_endpoint_cannot_redirect_or_embed_credentials(url):
    with pytest.raises(ValueError, match="LLM_BASE_URL"):
        Settings(provider="gemini", base_url=url)


def test_gemini_accepts_canonical_url_with_trailing_slash_and_custom_bare_model():
    settings = Settings(provider="gemini", base_url="https://generativelanguage.googleapis.com/v1beta/", model="gemini-3.5-flash-lite")
    assert settings.base_url == "https://generativelanguage.googleapis.com/v1beta"
    assert settings.model == "gemini-3.5-flash-lite"


@pytest.mark.parametrize("model", ["models/gemini-3.1-flash-lite", "../model", "gemini:generateContent", "gemini?key=value", "gemini\n"])
def test_gemini_model_cannot_inject_endpoint_paths(model):
    with pytest.raises(ValueError, match="Gemini LLM_MODEL"):
        Settings(provider="gemini", model=model)


def test_gemini_prefers_its_own_api_key_and_ignores_cloud_and_legacy_aliases(env_file):
    env_file.write_text("LLM_PROVIDER=gemini\nGEMINI_API_KEY=fictional-gemini-key\nGOOGLE_API_KEY=fictional-google-alias\nGOOGLE_CLOUD_API_KEY=fictional-cloud-key\nLLM_API_KEY=fictional-generic-key\nTOKENROUTER_API_KEY=fictional-tokenrouter\nOPENROUTER_API_KEY=fictional-openrouter\n", encoding="utf-8")
    settings = Settings.from_env(env_file)
    assert settings.api_key == "fictional-gemini-key"
    assert settings.model == "gemini-3.1-flash-lite"
    assert "fictional-gemini-key" not in repr(settings)
    assert set(settings.route_identity) == {"provider", "base_url"}


def test_gemini_google_api_key_alias_is_supported(env_file):
    env_file.write_text("LLM_PROVIDER=gemini\nGOOGLE_API_KEY=fictional-google-alias\n", encoding="utf-8")
    assert Settings.from_env(env_file).api_key == "fictional-google-alias"


@pytest.mark.parametrize("unrelated_name", ["GOOGLE_CLOUD_API_KEY", "LLM_API_KEY", "TOKENROUTER_API_KEY", "OPENROUTER_API_KEY"])
def test_gemini_never_implicitly_reuses_cloud_or_other_gateway_credentials(env_file, unrelated_name):
    env_file.write_text(f"LLM_PROVIDER=gemini\n{unrelated_name}=fictional-unrelated-key\n", encoding="utf-8")
    assert Settings.from_env(env_file).api_key == ""


def test_gemini_key_is_not_used_by_vertex_or_legacy_providers(env_file):
    env_file.write_text("LLM_PROVIDER=vertex\nGEMINI_API_KEY=fictional-gemini-key\n", encoding="utf-8")
    assert Settings.from_env(env_file).api_key == ""
    env_file.write_text("GEMINI_API_KEY=fictional-gemini-key\n", encoding="utf-8")
    assert Settings.from_env(env_file).api_key == ""
