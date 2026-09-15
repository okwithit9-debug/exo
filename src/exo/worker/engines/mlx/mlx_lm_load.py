"""mlx_lm load hooks used by the EXO MLX engine.

The primary Flash-Next target is
``orcarouter/Qwen3.8-Flash-Next-Uncensored-MLX``
(``Qwen4ExpForConditionalGeneration`` / ``qwen4_exp``). Stock mlx_lm does
not ship that module yet (ml-explore/mlx-lm#1788). EXO constructs the
orcarouter pack through:

1. Native ``mlx_lm.models.qwen4_exp`` when the pin has it.
2. A temporary mlx-vlm>=0.6.17 adapter
   (``src/exo/worker/engines/mlx/vendor/qwen4_exp.py``).
3. ``Qwen4ExpUnavailableError`` when neither is importable.

Packs that ship ``config.json`` ``model_file`` still load through mlx_lm
when that kwarg path exists. Qwen3.8-27B remains a secondary ``qwen3_5``
path through stock ``mlx_lm.models.qwen3_5``.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Final, cast

from exo.worker.engines.mlx.qwen4_exp_shim import (
    QWEN4_EXP_SHIM_INSTALL_MESSAGE,
    Qwen4ExpClassImporter,
    import_qwen4_exp_model_classes,
    install_qwen4_exp_shim_into_mlx_lm,
    read_mlx_model_config,
)

QWEN4_EXP_MODEL_TYPE: Final[str] = "qwen4_exp"
QWEN4_EXP_ARCHITECTURE_PREFIX: Final[str] = "Qwen4Exp"
QWEN4_EXP_UNAVAILABLE_MESSAGE: Final[str] = QWEN4_EXP_SHIM_INSTALL_MESSAGE


class Qwen4ExpUnavailableError(ValueError):
    """Raised when neither mlx_lm nor the mlx-vlm shim can construct qwen4_exp.

    Handled at runner load time (``load_mlx_items`` / ``shard_and_load``)
    so the user sees install / removal steps instead of a generic import
    error.
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
    import_qwen4_exp_classes: Qwen4ExpClassImporter | None = None,
) -> tuple[type[Any], type[Any]]:
    """Resolve mlx_lm Model / ModelArgs, falling back to the qwen4_exp shim.

    ``mlx_lm.utils.load_model`` calls this when the checkpoint has no
    ``model_file``. Native ``mlx_lm.models.qwen4_exp`` wins. Otherwise the
    mlx-vlm adapter is used until mlx-lm#1788 lands.
    """
    resolve_classes = get_classes
    if resolve_classes is None:
        try:
            mlx_lm_utils = importlib.import_module("mlx_lm.utils")
        except (ModuleNotFoundError, ImportError):
            mlx_lm_utils = None
        if mlx_lm_utils is not None:
            imported_get_classes = getattr(mlx_lm_utils, "_get_classes", None)
            if callable(imported_get_classes):
                resolve_classes = cast(
                    Callable[[dict[str, Any]], tuple[type[Any], type[Any]]],
                    imported_get_classes,
                )

    native_error: Exception | None = None
    if resolve_classes is not None:
        try:
            return resolve_classes(config)
        except (ValueError, ImportError, AttributeError, ModuleNotFoundError) as error:
            native_error = error
            if not is_qwen4_exp_config(config):
                raise

    if not is_qwen4_exp_config(config):
        raise ValueError("mlx_lm model class resolution is unavailable")

    load_shim_classes = (
        import_qwen4_exp_classes
        if import_qwen4_exp_classes is not None
        else import_qwen4_exp_model_classes
    )
    shim_classes = load_shim_classes()
    if shim_classes is None:
        raise Qwen4ExpUnavailableError(QWEN4_EXP_UNAVAILABLE_MESSAGE) from native_error
    return cast(tuple[type[Any], type[Any]], shim_classes)


def load_mlx_lm_model(
    model_path: Path,
    *,
    trust_remote_code: bool,
    load_model: Callable[..., tuple[object, dict[str, Any]]] | None = None,
    import_qwen4_exp_classes: Qwen4ExpClassImporter | None = None,
) -> tuple[object, dict[str, Any]]:
    """Load weights through mlx_lm, forwarding EXO hooks the pin supports.

    For Flash-Next, install the mlx-vlm shim into ``mlx_lm.models.qwen4_exp``
    when native classes are missing, and pass ``get_model_classes`` when the
    loader accepts it. Older mlx_lm forks omit ``trust_remote_code`` /
    ``get_model_classes``; those kwargs are only passed when present so
    Qwen3 / Qwen3.5 / Llama loads stay unchanged.
    """
    load_shim_classes = (
        import_qwen4_exp_classes
        if import_qwen4_exp_classes is not None
        else import_qwen4_exp_model_classes
    )
    disk_config = read_mlx_model_config(model_path)
    if disk_config is not None and is_qwen4_exp_config(disk_config):
        if load_shim_classes() is None:
            raise Qwen4ExpUnavailableError(QWEN4_EXP_UNAVAILABLE_MESSAGE)
        _ = install_qwen4_exp_shim_into_mlx_lm()

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
        if import_qwen4_exp_classes is None:
            load_parameters["get_model_classes"] = resolve_mlx_lm_model_classes
        else:

            def resolve_with_injected_shim(
                config: dict[str, Any],
            ) -> tuple[type[Any], type[Any]]:
                return resolve_mlx_lm_model_classes(
                    config,
                    import_qwen4_exp_classes=import_qwen4_exp_classes,
                )

            load_parameters["get_model_classes"] = resolve_with_injected_shim
    model, config = loader(model_path, **load_parameters)
    return model, config
