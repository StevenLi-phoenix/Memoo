"""Tests for AppConfig: save/load round-trip, display redaction, api_key preservation."""

from __future__ import annotations

import yaml

from core.config import AppConfig


class TestConfigRoundTrip:
    def test_loads_new_llm_config_shape(self, tmp_path) -> None:
        config_file = tmp_path / "config.yaml"
        initial = {
            "llm": {
                "default": "anthropic/claude-sonnet-4-6",
                "fallback": ["openai/gpt5.4-mini"],
                "compressor": "anthropic/claude-haiku-4-5",
                "providers": [
                    {
                        "name": "anthropic",
                        "provider": "anthropic",
                        "allow_model_discovery": True,
                    },
                    {
                        "name": "localhost",
                        "provider": "openai",
                        "base_url": "http://localhost:1234/v1",
                    },
                ],
                "models": [
                    {
                        "name": "anthropic/claude-sonnet-4-6",
                        "provider": "anthropic",
                        "model": "claude-sonnet-4-6",
                    },
                    {
                        "name": "localhost/gemma-4-e4b-it",
                        "provider": "localhost",
                        "model": "gemma-4-e4b-it",
                    },
                ],
            },
        }
        config_file.write_text(yaml.dump(initial), encoding="utf-8")

        cfg = AppConfig.load(str(config_file))

        assert cfg.llm.default == "anthropic/claude-sonnet-4-6"
        assert [p.name for p in cfg.llm.providers] == ["anthropic", "localhost"]
        assert cfg.llm.providers[0].allow_model_discovery is True
        assert cfg.llm.providers[1].base_url == "http://localhost:1234/v1"
        assert [m.name for m in cfg.llm.models] == [
            "anthropic/claude-sonnet-4-6",
            "localhost/gemma-4-e4b-it",
        ]
        assert cfg.llm.models[1].provider == "localhost"
        assert cfg.llm.models[1].model == "gemma-4-e4b-it"

    def test_save_preserves_embedding_api_key(self, tmp_path) -> None:
        """Regression: save() must not strip embedding.api_key from config.yaml."""
        config_file = tmp_path / "config.yaml"
        initial = {
            "embedding": {
                "provider": "openai",
                "base_url": "https://api.openai.com/v1",
                "model": "text-embedding-3-small",
                "api_key": "sk-secret-key-12345",
            },
        }
        config_file.write_text(yaml.dump(initial), encoding="utf-8")

        cfg = AppConfig.load(str(config_file))
        assert cfg.embedding.api_key == "sk-secret-key-12345"

        # Save (this triggers to_dict → YAML write)
        cfg.save()

        # Reload and verify api_key survived
        reloaded = AppConfig.load(str(config_file))
        assert reloaded.embedding.api_key == "sk-secret-key-12345"

    def test_save_preserves_all_sections(self, tmp_path) -> None:
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml.dump({"port": 9999}), encoding="utf-8")

        cfg = AppConfig.load(str(config_file))
        cfg.port = 9999
        cfg.save()

        reloaded = AppConfig.load(str(config_file))
        assert reloaded.port == 9999

    def test_save_preserves_llm_models(self, tmp_path) -> None:
        config_file = tmp_path / "config.yaml"
        initial = {
            "llm": {
                "default": "localhost/gemma-4-e4b-it",
                "providers": [
                    {
                        "name": "localhost",
                        "provider": "openai",
                        "base_url": "http://localhost:1234/v1",
                    },
                ],
                "models": [
                    {
                        "name": "localhost/gemma-4-e4b-it",
                        "provider": "localhost",
                        "model": "gemma-4-e4b-it",
                    },
                ],
            },
        }
        config_file.write_text(yaml.dump(initial), encoding="utf-8")

        cfg = AppConfig.load(str(config_file))
        cfg.save()

        reloaded = AppConfig.load(str(config_file))
        assert reloaded.llm.default == "localhost/gemma-4-e4b-it"
        assert len(reloaded.llm.providers) == 1
        assert len(reloaded.llm.models) == 1
        assert reloaded.llm.models[0].name == "localhost/gemma-4-e4b-it"

    def test_save_preserves_explicit_allow_model_discovery_false(self, tmp_path) -> None:
        config_file = tmp_path / "config.yaml"
        initial = {
            "llm": {
                "providers": [
                    {
                        "name": "openai",
                        "provider": "openai",
                        "allow_model_discovery": False,
                    },
                ],
                "models": [
                    {
                        "name": "openai/gpt5.4-mini",
                        "provider": "openai",
                        "model": "gpt5.4-mini",
                    },
                ],
            },
        }
        config_file.write_text(yaml.dump(initial), encoding="utf-8")

        cfg = AppConfig.load(str(config_file))
        assert cfg.llm.providers[0].allow_model_discovery is False

        cfg.save()

        raw = yaml.safe_load(config_file.read_text(encoding="utf-8"))
        provider = raw["llm"]["providers"][0]
        assert "allow_model_discovery" in provider
        assert provider["allow_model_discovery"] is False


class TestDisplayDict:
    def test_api_key_redacted_in_display(self) -> None:
        cfg = AppConfig()
        cfg.embedding.api_key = "sk-secret-12345"

        display = cfg.to_display_dict()
        assert display["embedding"]["api_key"] == "***"

    def test_empty_api_key_not_redacted(self) -> None:
        cfg = AppConfig()
        cfg.embedding.api_key = ""

        display = cfg.to_display_dict()
        # Empty string is falsy, so no redaction applied
        assert display["embedding"]["api_key"] == ""

    def test_to_dict_includes_real_key(self) -> None:
        cfg = AppConfig()
        cfg.embedding.api_key = "sk-real-key"

        full = cfg.to_dict()
        assert full["embedding"]["api_key"] == "sk-real-key"

    def test_display_and_dict_are_independent(self) -> None:
        """to_display_dict should not mutate the original config."""
        cfg = AppConfig()
        cfg.embedding.api_key = "sk-original"

        _ = cfg.to_display_dict()
        # Original config should be untouched
        assert cfg.embedding.api_key == "sk-original"
        assert cfg.to_dict()["embedding"]["api_key"] == "sk-original"
