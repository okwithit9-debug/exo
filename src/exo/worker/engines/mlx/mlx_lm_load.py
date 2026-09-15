"""mlx_lm load hooks used by the EXO MLX engine.

The primary Flash-Next target is
``orcarouter/Qwen3.8-Flash-Next-Uncensored-MLX``
(``Qwen4ExpForConditionalGeneration`` / ``qwen4_exp``). That architecture
is not in released mlx_lm (see ml-explore/mlx-lm#1788). Packs that ship a
``model_file`` (typically ``qwen4_exp.py``) can import via
``load_model(..., trust_remote_code=True)``. The orcarouter MLX pack is
converted with mlx-vlm and needs either vendored ``qwen4_exp`` or a
``model_file`` shim.

mlx-vlm >= 0.6.17 can load Flash-Next as a standalone VLM (PR #2032) but
is not wired into EXO's mlx_lm disaggregation path.

Qwen3.8-27B remains a secondary ``qwen3_5`` path through stock
``mlx_lm.models.qwen3_5``.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Final, cast

QWEN4_EXP_MODEL_TYPE: Final[str] = "qwen4_exp"
QWEN4_EXP_ARCHITECTURE_PREFIX: Final[str] = "Qwen4Exp"

QWEN4_EXP_UNAVAILABLE_MESSAGE: Final[str] = (
    "This checkpoint uses the qwen4_exp / Qwen4Exp* architecture "
    "(orcarouter/Qwen3.8-Flash-Next-Uncensored-MLX). Stock mlx_lm "
    "cannot construct it yet (unmerged "
    "https://github.com/ml-explore/mlx-lm/pull/1788). "
    "Next step: pin an mlx_lm build that vendors mlx_lm.models.qwen4_exp, "
    "or add config.json model_file (for example qwen4_exp.py) with "
    "trust_remote_code=true on the model card. "
    "Standalone mlx-vlm>=0.6.17 can load this pack outside EXO; "
    "Mac+Spark EXO disaggregation still needs the mlx_lm module so "
    "auto_parallel can see typed Qwen4Exp layers. "
    "orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4 is the Spark vLLM "
    "companion, not an EXO MLX weight."
)


class Qwen4ExpUnavailableError(ValueError):
    """Raised when mlx_lm has no qwen4_exp implementation and no model_file.

    Handled at runner load time (``load_mlx_items`` / ``shard_and_load``)
    so the user sees a concrete next step instead of a generic import error.
    """


def is_qwen4_exp_config(config: Mapping[str, object]) -> bool:
    """Return True when a HuggingFace / MLX config.json is Flash-Next."""
    model_type = config.get("model_type")
    text_config = config.get("text_config")
    text_model_type: str | None = None
    if isinstance(text_config, dict):
        raw_text_model_type = text_config.get("model_type")  # pyright: ignore[reportUnknownVariableType,reportUnknownMemberType]
        if isinstance(raw_text_model_type, str):
            text_model_type = raw_text_model_type
    architectures = config.get("architectures")
    architecture_names: list[str] = []
    if isinstance(architectures, list):
        for name in architectures:  # pyright: ignore[reportUnknownVariableType]
            if isinstance(name, str):
                architecture_names.append(name)
    return (
        model_type == QWEN4_EXP_MODEL_TYPE
        or text_model_type == QWEN4_EXP_MODEL_TYPE
        or any(
            name.startswith(QWEN4_EXP_ARCHITECTURE_PREFIX)
            for name in architecture_names
        )
    )


def resolve_mlx_lm_model_classes(
    config: dict[str, Any],
    *,
    get_classes: Callable[[dict[str, Any]], tuple[type[Any], type[Any]]] | None = None,
) -> tuple[type[Any], type[Any]]:
    """Resolve mlx_lm Model / ModelArgs, with a Flash-Next-specific error.

    ``mlx_lm.utils.load_model`` calls this when the checkpoint has no
    ``model_file``. If mlx_lm later vendors ``qwen4_exp``, ``_get_classes``
    succeeds and EXO needs no further change on the construct path.
    """
    resolve_classes = get_classes
    if resolve_classes is None:
        mlx_lm_utils = importlib.import_module("mlx_lm.utils")
        imported_get_classes = getattr(mlx_lm_utils, "_get_classes", None)
        if not callable(imported_get_classes):
            raise Qwen4ExpUnavailableError(QWEN4_EXP_UNAVAILABLE_MESSAGE)
        resolve_classes = cast(
            Callable[[dict[str, Any]], tuple[type[Any], type[Any]]],
            imported_get_classes,
        )

    try:
        return resolve_classes(config)
    except (ValueError, ImportError, AttributeError, ModuleNotFoundError) as error:
        if not is_qwen4_exp_config(config):
            raise
        raise Qwen4ExpUnavailableError(QWEN4_EXP_UNAVAILABLE_MESSAGE) from error


def load_mlx_lm_model(
    model_path: Path,
    *,
    trust_remote_code: bool,
    load_model: Callable[..., tuple[object, dict[str, Any]]] | None = None,
) -> tuple[object, dict[str, Any]]:
    """Load weights through mlx_lm, forwarding EXO hooks the pin supports.

    Older mlx_lm forks (including some EXO pins) omit
    ``trust_remote_code`` / ``get_model_classes``. Those kwargs are only
    passed when present so Qwen3 / Qwen3.5 / Llama loads stay unchanged.
    """
    loader = load_model
    if loader is None:
        mlx_lm_utils = importlib.import_module("mlx_lm.utils")
        loader = cast(
            Callable[..., tuple[object, dict[str, Any]]],
            mlx_lm_utils.load_model,
        )

    load_parameters: dict[str, Any] = {"lazy": True, "strict": False}
    signature = inspect.signature(loader)
    if "trust_remote_code" in signature.parameters:
        load_parameters["trust_remote_code"] = trust_remote_code
    if "get_model_classes" in signature.parameters:
        load_parameters["get_model_classes"] = resolve_mlx_lm_model_classes
    model, config = loader(model_path, **load_parameters)
    return model, config
