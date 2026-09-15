from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Protocol, cast

import pytest

from exo.shared.types.common import ModelId
from exo.worker.engines.mlx.eos_tokens import get_eos_token_ids_for_model
from exo.worker.engines.mlx.mlx_lm_load import (
    QWEN4_EXP_UNAVAILABLE_MESSAGE,
    Qwen4ExpUnavailableError,
    is_qwen4_exp_config,
    load_mlx_lm_model,
    resolve_mlx_lm_model_classes,
)
from exo.worker.engines.mlx.qwen4_exp_shim import (
    MLX_LM_QWEN4_EXP_MODULE,
    build_mlx_lm_compatible_qwen4_exp_classes,
    detect_qwen4_exp_implementation_source,
    install_qwen4_exp_shim_into_mlx_lm,
    is_vision_weight_key,
    module_is_exo_qwen4_exp_shim,
    read_mlx_model_config,
)


def test_resolve_uses_mlx_lm_when_qwen3_5_is_registered() -> None:
    sentinel_model = type("Model", (), {})
    sentinel_args = type("ModelArgs", (), {})

    def get_classes(
        config: dict[str, Any],
    ) -> tuple[type[Any], type[Any]]:
        assert config["model_type"] == "qwen3_5"
        return sentinel_model, sentinel_args

    model_class, args_class = resolve_mlx_lm_model_classes(
        {
            "model_type": "qwen3_5",
            "architectures": ["Qwen3_5ForConditionalGeneration"],
        },
        get_classes=get_classes,
    )
    assert model_class is sentinel_model
    assert args_class is sentinel_args


def test_resolve_raises_documented_error_when_shim_missing() -> None:
    def get_classes(_config: dict[str, Any]) -> tuple[type[Any], type[Any]]:
        raise ValueError("Model type qwen4_exp not supported.")

    with pytest.raises(Qwen4ExpUnavailableError, match="ml-explore/mlx-lm/pull/1788"):
        resolve_mlx_lm_model_classes(
            {
                "model_type": "qwen4_exp",
                "architectures": ["Qwen4ExpForConditionalGeneration"],
            },
            get_classes=get_classes,
            import_qwen4_exp_classes=lambda: None,
        )


def test_resolve_does_not_rewrite_unknown_non_flash_next_errors() -> None:
    def get_classes(_config: dict[str, Any]) -> tuple[type[Any], type[Any]]:
        raise ValueError("Model type llama-unknown not supported.")

    with pytest.raises(ValueError, match="llama-unknown"):
        resolve_mlx_lm_model_classes(
            {"model_type": "llama-unknown"},
            get_classes=get_classes,
        )


def test_load_mlx_lm_model_forwards_hooks_when_supported() -> None:
    def fake_load_model(
        model_path: Path,
        *,
        lazy: bool = False,
        strict: bool = True,
        trust_remote_code: bool = False,
        get_model_classes: object | None = None,
    ) -> tuple[object, dict[str, Any]]:
        assert lazy is True
        assert strict is False
        assert trust_remote_code is True
        assert get_model_classes is resolve_mlx_lm_model_classes
        assert model_path == Path("/tmp/qwen38")
        return "model", {"model_type": "qwen3_5"}

    loaded_model, config = load_mlx_lm_model(
        Path("/tmp/qwen38"),
        trust_remote_code=True,
        load_model=fake_load_model,
    )
    assert loaded_model == "model"
    assert config["model_type"] == "qwen3_5"


def test_load_mlx_lm_model_skips_unknown_kwargs_on_old_mlx_lm() -> None:
    captured: dict[str, Any] = {}

    def old_load_model(
        model_path: Path, lazy: bool = False, strict: bool = True
    ) -> tuple[object, dict[str, str]]:
        captured["lazy"] = lazy
        captured["strict"] = strict
        captured["path"] = model_path
        return ("old-model", {"model_type": "llama"})

    assert "trust_remote_code" not in inspect.signature(old_load_model).parameters
    loaded_model, config = load_mlx_lm_model(
        Path("/tmp/llama"),
        trust_remote_code=True,
        load_model=old_load_model,
    )
    assert loaded_model == "old-model"
    assert config["model_type"] == "llama"
    assert captured == {
        "lazy": True,
        "strict": False,
        "path": Path("/tmp/llama"),
    }


def test_qwen4_exp_error_message_is_actionable() -> None:
    assert (
        "orcarouter/Qwen3.8-Flash-Next-Uncensored-MLX" in QWEN4_EXP_UNAVAILABLE_MESSAGE
    )
    assert "mlx-vlm>=0.6.17" in QWEN4_EXP_UNAVAILABLE_MESSAGE
    assert "vendor/qwen4_exp.py" in QWEN4_EXP_UNAVAILABLE_MESSAGE
    assert "qwen4_exp_shim.py" in QWEN4_EXP_UNAVAILABLE_MESSAGE
    assert "NVFP4" in QWEN4_EXP_UNAVAILABLE_MESSAGE


@pytest.mark.parametrize(
    "model_id",
    [
        "orcarouter/Qwen3.8-Flash-Next-Uncensored-MLX",
        "mlx-community/Qwen3.8-27B-4bit",
        "qwen-3.8-flash-next",
        "qwen-3.8-27b",
    ],
)
def test_qwen38_eos_token_ids(model_id: str) -> None:
    assert get_eos_token_ids_for_model(ModelId(model_id)) == [248046, 248044]


def test_qwen3_and_llama_eos_paths_unchanged() -> None:
    assert get_eos_token_ids_for_model(ModelId("mlx-community/Qwen3-0.6B-4bit")) is None
    assert (
        get_eos_token_ids_for_model(
            ModelId("mlx-community/llama-3.3-70b-instruct-fp16")
        )
        is None
    )
    assert get_eos_token_ids_for_model(ModelId("mlx-community/Qwen3.5-27B-4bit")) == [
        248046,
        248044,
    ]


def test_flash_next_config_detection() -> None:
    assert (
        is_qwen4_exp_config(
            {
                "architectures": ["Qwen4ExpForConditionalGeneration"],
                "model_type": "qwen4_exp",
            }
        )
        is True
    )
    assert (
        is_qwen4_exp_config(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "model_type": "qwen3_5",
            }
        )
        is False
    )
    assert is_qwen4_exp_config({"architectures": ["Qwen4ExpForCausalLM"]}) is True
    assert is_qwen4_exp_config({"model_type": "llama"}) is False


def _flash_next_config() -> dict[str, Any]:
    return {
        "model_type": "qwen4_exp",
        "architectures": ["Qwen4ExpForConditionalGeneration"],
    }


def test_resolve_prefers_native_qwen4_exp_when_registered() -> None:
    native_model = type("NativeModel", (), {})
    native_args = type("NativeArgs", (), {})

    def get_classes(
        config: dict[str, Any],
    ) -> tuple[type[Any], type[Any]]:
        assert config["model_type"] == "qwen4_exp"
        return native_model, native_args

    def unused_shim() -> tuple[type[Any], type[Any]] | None:
        raise AssertionError("shim must not run when mlx_lm resolves qwen4_exp")

    model_class, args_class = resolve_mlx_lm_model_classes(
        _flash_next_config(),
        get_classes=get_classes,
        import_qwen4_exp_classes=unused_shim,
    )
    assert model_class is native_model
    assert args_class is native_args


def test_resolve_falls_back_to_shim_when_mlx_lm_missing_qwen4_exp() -> None:
    shim_model = type("ShimModel", (), {})
    shim_args = type("ShimModelArgs", (), {})

    def get_classes(_config: dict[str, Any]) -> tuple[type[Any], type[Any]]:
        raise ValueError("Model type qwen4_exp not supported.")

    model_class, args_class = resolve_mlx_lm_model_classes(
        _flash_next_config(),
        get_classes=get_classes,
        import_qwen4_exp_classes=lambda: (shim_model, shim_args),
    )
    assert model_class is shim_model
    assert args_class is shim_args


def test_detect_prefers_native_mlx_lm_over_mlx_vlm() -> None:
    native = ModuleType("mlx_lm.models.qwen4_exp")
    mlx_vlm = ModuleType("mlx_vlm.models.qwen4_exp")
    assert (
        detect_qwen4_exp_implementation_source(
            import_mlx_lm_module=lambda: native,
            import_mlx_vlm_module=lambda: mlx_vlm,
        )
        == "mlx_lm"
    )


def test_detect_falls_back_to_mlx_vlm() -> None:
    mlx_vlm = ModuleType("mlx_vlm.models.qwen4_exp")
    assert (
        detect_qwen4_exp_implementation_source(
            import_mlx_lm_module=lambda: None,
            import_mlx_vlm_module=lambda: mlx_vlm,
        )
        == "mlx_vlm"
    )


def test_detect_returns_none_when_shim_missing() -> None:
    assert (
        detect_qwen4_exp_implementation_source(
            import_mlx_lm_module=lambda: None,
            import_mlx_vlm_module=lambda: None,
        )
        is None
    )


class _ShimMarkerModule(ModuleType):
    EXO_QWEN4_EXP_SHIM: bool


class _ConstructedShim(Protocol):
    layers: object

    def make_cache(self) -> object: ...

    def __call__(
        self,
        tokens: object,
        cache: object = None,
        input_embeddings: object = None,
    ) -> object: ...

    def sanitize(self, weights: dict[str, object]) -> dict[str, object]: ...


class _ModelArgsLike(Protocol):
    @staticmethod
    def from_dict(params: dict[str, Any]) -> object: ...


def test_install_registers_shim_without_overwriting_native() -> None:
    shim = _ShimMarkerModule("exo.worker.engines.mlx.vendor.qwen4_exp")
    shim.EXO_QWEN4_EXP_SHIM = True
    registry: dict[str, ModuleType] = {}

    source = install_qwen4_exp_shim_into_mlx_lm(
        module_registry=registry,
        import_mlx_lm_module=lambda: None,
        import_shim_module=lambda: shim,
    )
    assert source == "mlx_vlm"
    assert registry[MLX_LM_QWEN4_EXP_MODULE] is shim
    assert module_is_exo_qwen4_exp_shim(shim) is True

    native = ModuleType("mlx_lm.models.qwen4_exp")
    guarded: dict[str, ModuleType] = {}
    assert (
        install_qwen4_exp_shim_into_mlx_lm(
            module_registry=guarded,
            import_mlx_lm_module=lambda: native,
            import_shim_module=lambda: shim,
        )
        == "mlx_lm"
    )
    assert guarded == {}


def test_shim_classes_construct_and_return_logits() -> None:
    class DummyModule:
        def __init__(self) -> None:
            pass

    class DummyLanguageModel:
        def __init__(self, text_config: object, config: object) -> None:
            self.text_config = text_config
            self.config = config
            self.model = SimpleNamespace(layers=["layer-0", "layer-1"])

        def __call__(
            self,
            tokens: object,
            inputs_embeds: object = None,
            cache: object = None,
        ) -> SimpleNamespace:
            assert tokens == "TOKENS"
            assert cache == "CACHE"
            assert inputs_embeds == "EMBEDS"
            return SimpleNamespace(logits="LOGITS")

        def make_cache(self) -> list[str]:
            return ["gdn", "qsa"]

    class DummyConfig:
        def __init__(self) -> None:
            self.model_type = "qwen4_exp"
            self.text_config = SimpleNamespace(hidden_size=2560)

        @classmethod
        def from_dict(cls, params: dict[str, Any]) -> DummyConfig:
            loaded = cls()
            raw_model_type = cast(object, params.get("model_type", "qwen4_exp"))
            loaded.model_type = str(raw_model_type)
            return loaded

    def fake_sanitize(_host: object, weights: dict[str, object]) -> dict[str, object]:
        return {
            **weights,
            "language_model.model.layers.0.weight": 1,
            "vision_tower.patch_embed.weight": 2,
            "sanitized": True,
        }

    model_class, args_class = build_mlx_lm_compatible_qwen4_exp_classes(
        language_model_factory=DummyLanguageModel,
        model_config_type=DummyConfig,
        sanitize_weights=fake_sanitize,
        module_type=DummyModule,
    )
    model_args: object = cast(_ModelArgsLike, args_class).from_dict(
        _flash_next_config()
    )
    constructed = cast(Callable[[object], object], model_class)
    model = cast(_ConstructedShim, constructed(model_args))
    assert model.layers == ["layer-0", "layer-1"]
    assert model.make_cache() == ["gdn", "qsa"]
    assert model("TOKENS", cache="CACHE", input_embeddings="EMBEDS") == "LOGITS"
    sanitized = model.sanitize(
        {
            "mtp.draft.weight": 0,
            "language_model.model.layers.0.weight": 1,
            "vision_tower.patch_embed.weight": 2,
        }
    )
    assert isinstance(sanitized, dict)
    assert "vision_tower.patch_embed.weight" not in sanitized
    assert sanitized["language_model.model.layers.0.weight"] == 1
    assert sanitized["sanitized"] is True


def test_vision_weight_key_filter() -> None:
    assert is_vision_weight_key("vision_tower.blocks.0.weight") is True
    assert is_vision_weight_key("language_model.model.layers.0.weight") is False


def test_load_raises_documented_error_when_flash_next_shim_missing(
    tmp_path: Path,
) -> None:
    _ = (tmp_path / "config.json").write_text(
        json.dumps(_flash_next_config()),
        encoding="utf-8",
    )

    def unused_loader(
        _model_path: Path, **_kwargs: object
    ) -> tuple[object, dict[str, Any]]:
        raise AssertionError("load_model should not run when the shim is missing")

    with pytest.raises(Qwen4ExpUnavailableError, match="mlx-vlm>=0.6.17"):
        load_mlx_lm_model(
            tmp_path,
            trust_remote_code=True,
            load_model=unused_loader,
            import_qwen4_exp_classes=lambda: None,
        )


def test_load_forwards_resolver_when_flash_next_shim_present(
    tmp_path: Path,
) -> None:
    _ = (tmp_path / "config.json").write_text(
        json.dumps(_flash_next_config()),
        encoding="utf-8",
    )
    shim_model = type("ShimModel", (), {})
    shim_args = type("ShimArgs", (), {})
    captured: dict[str, object] = {}

    def fake_load_model(
        model_path: Path,
        *,
        lazy: bool = False,
        strict: bool = True,
        get_model_classes: object | None = None,
    ) -> tuple[object, dict[str, Any]]:
        captured["path"] = model_path
        captured["lazy"] = lazy
        captured["strict"] = strict
        captured["get_model_classes"] = get_model_classes
        assert callable(get_model_classes)
        resolver = cast(
            Callable[[dict[str, Any]], tuple[type[object], type[object]]],
            get_model_classes,
        )
        model_class, args_class = resolver(_flash_next_config())
        assert model_class is shim_model
        assert args_class is shim_args
        return "constructed", _flash_next_config()

    loaded_model, config = load_mlx_lm_model(
        tmp_path,
        trust_remote_code=True,
        load_model=fake_load_model,
        import_qwen4_exp_classes=lambda: (shim_model, shim_args),
    )
    assert loaded_model == "constructed"
    assert config["model_type"] == "qwen4_exp"
    assert captured["path"] == tmp_path
    assert captured["lazy"] is True
    assert captured["strict"] is False


def test_read_mlx_model_config_roundtrip(tmp_path: Path) -> None:
    payload = _flash_next_config()
    _ = (tmp_path / "config.json").write_text(json.dumps(payload), encoding="utf-8")
    assert read_mlx_model_config(tmp_path) == payload
    assert read_mlx_model_config(tmp_path / "missing") is None
