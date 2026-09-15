import inspect
from pathlib import Path
from typing import Any

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


def test_resolve_raises_documented_error_for_qwen4_exp() -> None:
    def get_classes(_config: dict[str, Any]) -> tuple[type[Any], type[Any]]:
        raise ValueError("Model type qwen4_exp not supported.")

    with pytest.raises(Qwen4ExpUnavailableError, match="ml-explore/mlx-lm/pull/1788"):
        resolve_mlx_lm_model_classes(
            {
                "model_type": "qwen4_exp",
                "architectures": ["Qwen4ExpForConditionalGeneration"],
            },
            get_classes=get_classes,
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
    assert "model_file" in QWEN4_EXP_UNAVAILABLE_MESSAGE
    assert "auto_parallel" in QWEN4_EXP_UNAVAILABLE_MESSAGE


@pytest.mark.parametrize(
    "model_id",
    [
        "mlx-community/Qwen3.8-27B-4bit",
        "local/Qwen3.8-27B-Uncensored-MLX-4bit",
        "mlx-community/Qwen3.8-Flash-Next-4bit",
        "qwen-3.8-27b",
    ],
)
def test_qwen38_eos_token_ids(model_id: str) -> None:
    assert get_eos_token_ids_for_model(ModelId(model_id)) == [248046, 248044]


def test_qwen3_and_llama_eos_paths_unchanged() -> None:
    assert get_eos_token_ids_for_model(ModelId("mlx-community/Qwen3-0.6B-4bit")) is None
    assert (
        get_eos_token_ids_for_model(ModelId("mlx-community/llama-3.3-70b-instruct-fp16"))
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
