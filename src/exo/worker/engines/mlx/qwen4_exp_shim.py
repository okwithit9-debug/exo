"""Temporary qwen4_exp construct path until mlx-lm vendors the module.

Resolution order for ``orcarouter/Qwen3.8-Flash-Next-Uncensored-MLX``:

1. Native ``mlx_lm.models.qwen4_exp`` (ml-explore/mlx-lm#1788 or equivalent).
2. EXO adapter around ``mlx_vlm.models.qwen4_exp`` (mlx-vlm>=0.6.17).
3. ``Qwen4ExpUnavailableError`` with install / removal instructions.

The mlx-vlm adapter exists only so ``mlx_lm.utils.load_model`` can construct
the orcarouter Uncensored-MLX pack. It does not add typed Qwen4Exp
tensor/pipeline parallelism, and it does not prove a full MoE / GDN / QSA
/ PLE generate pass.

Delete ``vendor/qwen4_exp.py`` and the mlx-vlm branch below when native
``mlx_lm.models.qwen4_exp`` is on the EXO mlx-lm pin.
"""

from __future__ import annotations

import importlib
import json
import sys
from collections.abc import Callable, MutableMapping
from pathlib import Path
from types import ModuleType
from typing import Any, Final, Literal, Protocol, cast

Qwen4ExpImplementationSource = Literal["mlx_lm", "mlx_vlm"]
QWEN4_EXP_MODEL_TYPE: Final[str] = "qwen4_exp"

EXO_QWEN4_EXP_SHIM_ATTRIBUTE: Final[str] = "EXO_QWEN4_EXP_SHIM"
MLX_LM_QWEN4_EXP_MODULE: Final[str] = "mlx_lm.models.qwen4_exp"
MLX_VLM_QWEN4_EXP_PACKAGE: Final[str] = "mlx_vlm.models.qwen4_exp"
EXO_VENDOR_QWEN4_EXP_MODULE: Final[str] = "exo.worker.engines.mlx.vendor.qwen4_exp"

QWEN4_EXP_SHIM_INSTALL_MESSAGE: Final[str] = (
    "This checkpoint uses the qwen4_exp / Qwen4Exp* architecture "
    "(orcarouter/Qwen3.8-Flash-Next-Uncensored-MLX). Stock mlx_lm "
    "cannot construct it yet (unmerged "
    "https://github.com/ml-explore/mlx-lm/pull/1788). "
    "Install the EXO mlx extra and mlx-vlm>=0.6.17 so EXO can use the "
    "temporary mlx-vlm-backed shim "
    "(src/exo/worker/engines/mlx/vendor/qwen4_exp.py). "
    "Remove that vendor module and the mlx_vlm fallback in "
    "qwen4_exp_shim.py once import mlx_lm.models.qwen4_exp works on the "
    "EXO mlx-lm pin. "
    "orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4 is the Spark vLLM "
    "companion, not an EXO MLX weight."
)

_VISION_WEIGHT_ROOTS: Final[frozenset[str]] = frozenset(
    {
        "vision_tower",
        "visual",
        "vision_model",
        "multi_modal_projector",
    }
)

Qwen4ExpClassPair = tuple[type[object], type[object]]
Qwen4ExpClassImporter = Callable[[], Qwen4ExpClassPair | None]
LanguageModelFactory = Callable[[object, object], object]
ModuleImporter = Callable[[], ModuleType | None]
SanitizeWeights = Callable[[object, dict[str, object]], dict[str, object]]


class _LanguageModelLike(Protocol):
    model: object

    def __call__(
        self,
        tokens: object,
        inputs_embeds: object = None,
        cache: object = None,
    ) -> object: ...

    def make_cache(self) -> object: ...


class _LayerContainer(Protocol):
    layers: object


def is_vision_weight_key(key: str) -> bool:
    """Return True for vision-tower keys the text engine should not load."""
    root_name = key.split(".", 1)[0]
    return root_name in _VISION_WEIGHT_ROOTS


def module_is_exo_qwen4_exp_shim(module: ModuleType) -> bool:
    """Return True when a module is EXO's temporary mlx-vlm adapter."""
    return getattr(module, EXO_QWEN4_EXP_SHIM_ATTRIBUTE, False) is True


def read_mlx_model_config(model_path: Path) -> dict[str, Any] | None:
    """Read config.json if present. Returns None when the file is unreadable.

    Used by ``load_mlx_lm_model`` to fail Fast-Next packs with a documented
    error before mlx_lm raises a generic import error. Callers treat None as
    "defer to the loader".
    """
    config_path = model_path / "config.json"
    try:
        raw_text = config_path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        loaded_object: object = json.loads(raw_text)  # pyright: ignore[reportAny]
    except json.JSONDecodeError:
        return None
    if not isinstance(loaded_object, dict):
        return None
    typed_config: dict[str, Any] = {}
    loaded_mapping = cast(dict[object, object], loaded_object)
    for raw_key, raw_value in loaded_mapping.items():
        typed_config[str(raw_key)] = raw_value
    return typed_config


def _import_optional_module(module_name: str) -> ModuleType | None:
    try:
        return importlib.import_module(module_name)
    except (ModuleNotFoundError, ImportError):
        return None


def import_native_mlx_lm_qwen4_exp_module() -> ModuleType | None:
    """Import stock mlx_lm qwen4_exp, ignoring EXO's installed shim."""
    module = _import_optional_module(MLX_LM_QWEN4_EXP_MODULE)
    if module is None or module_is_exo_qwen4_exp_shim(module):
        return None
    return module


def import_mlx_vlm_qwen4_exp_module() -> ModuleType | None:
    return _import_optional_module(MLX_VLM_QWEN4_EXP_PACKAGE)


def detect_qwen4_exp_implementation_source(
    *,
    import_mlx_lm_module: ModuleImporter | None = None,
    import_mlx_vlm_module: ModuleImporter | None = None,
) -> Qwen4ExpImplementationSource | None:
    """Return which qwen4_exp implementation is available, if any."""
    load_mlx_lm = (
        import_mlx_lm_module
        if import_mlx_lm_module is not None
        else import_native_mlx_lm_qwen4_exp_module
    )
    load_mlx_vlm = (
        import_mlx_vlm_module
        if import_mlx_vlm_module is not None
        else import_mlx_vlm_qwen4_exp_module
    )
    if load_mlx_lm() is not None:
        return "mlx_lm"
    if load_mlx_vlm() is not None:
        return "mlx_vlm"
    return None


def _class_pair_from_module(module: ModuleType) -> Qwen4ExpClassPair | None:
    model_class = getattr(module, "Model", None)
    model_args_class = getattr(module, "ModelArgs", None)
    if isinstance(model_class, type) and isinstance(model_args_class, type):
        return cast(type[object], model_class), cast(type[object], model_args_class)
    return None


def import_native_mlx_lm_qwen4_exp_classes() -> Qwen4ExpClassPair | None:
    module = import_native_mlx_lm_qwen4_exp_module()
    if module is None:
        return None
    return _class_pair_from_module(module)


def build_mlx_lm_compatible_qwen4_exp_classes(
    *,
    language_model_factory: LanguageModelFactory,
    model_config_type: type[object],
    sanitize_weights: SanitizeWeights,
    module_type: type[object],
) -> Qwen4ExpClassPair:
    """Wrap mlx-vlm LanguageModel so mlx_lm construct + EXO generate agree.

    mlx_lm calls ``ModelArgs.from_dict(config)`` then ``Model(args)``.
    EXO's generate loop expects ``__call__(tokens, cache, input_embeddings)``
    to return logits (``mx.array``), not mlx-vlm ``LanguageModelOutput``.
    The returned Model subclasses ``module_type`` (``mlx.nn.Module`` in
    production) so ``load_weights`` can see ``language_model`` children.
    """

    class Qwen4ExpMlxLmModel:
        def __init__(self, config: object) -> None:
            super().__init__()
            self.config: object = config
            self.args: object = config
            self.model_type: object = getattr(
                config, "model_type", QWEN4_EXP_MODEL_TYPE
            )
            text_config: object = getattr(config, "text_config", config)
            self.language_model: _LanguageModelLike = cast(
                _LanguageModelLike,
                language_model_factory(text_config, config),
            )

        def sanitize(self, weights: dict[str, object]) -> dict[str, object]:
            remapped = sanitize_weights(self, weights)
            return {
                key: value
                for key, value in remapped.items()
                if not is_vision_weight_key(key)
            }

        def __call__(
            self,
            tokens: object,
            cache: object = None,
            input_embeddings: object = None,
        ) -> object:
            output_object: object = self.language_model(
                tokens,
                inputs_embeds=input_embeddings,
                cache=cache,
            )
            logits: object = getattr(output_object, "logits", output_object)
            return logits

        def make_cache(self) -> object:
            return self.language_model.make_cache()

        @property
        def model(self) -> object:
            inner: object = self.language_model.model
            if inner is None:
                raise AttributeError("Qwen4Exp language model has no inner model")
            return inner

        @property
        def layers(self) -> object:
            return cast(_LayerContainer, self.model).layers

        @property
        def quant_predicate(self) -> object:
            return getattr(self.language_model, "quant_predicate", True)

        @property
        def cast_predicate(self) -> object:
            return getattr(self.language_model, "cast_predicate", True)

    combined = type(
        "Qwen4ExpMlxLmModel",
        (Qwen4ExpMlxLmModel, module_type),
        {},
    )
    return cast(type[object], combined), model_config_type


def import_mlx_vlm_backed_qwen4_exp_classes() -> Qwen4ExpClassPair | None:
    """Build mlx_lm-compatible classes from mlx-vlm>=0.6.17 qwen4_exp."""
    language_module = _import_optional_module("mlx_vlm.models.qwen4_exp.language")
    config_module = _import_optional_module("mlx_vlm.models.qwen4_exp.config")
    model_module = _import_optional_module("mlx_vlm.models.qwen4_exp.qwen4_exp")
    mlx_nn_module = _import_optional_module("mlx.nn")
    if (
        language_module is None
        or config_module is None
        or model_module is None
        or mlx_nn_module is None
    ):
        return None

    language_model_type = getattr(language_module, "LanguageModel", None)
    model_config_type = getattr(config_module, "ModelConfig", None)
    vision_language_model_type = getattr(model_module, "Model", None)
    module_type = getattr(mlx_nn_module, "Module", None)
    sanitize = getattr(vision_language_model_type, "sanitize", None)
    if (
        not isinstance(language_model_type, type)
        or not isinstance(model_config_type, type)
        or not isinstance(module_type, type)
        or not callable(sanitize)
    ):
        return None

    return build_mlx_lm_compatible_qwen4_exp_classes(
        language_model_factory=cast(LanguageModelFactory, language_model_type),
        model_config_type=cast(type[object], model_config_type),
        sanitize_weights=cast(SanitizeWeights, sanitize),
        module_type=cast(type[object], module_type),
    )


def import_qwen4_exp_model_classes() -> Qwen4ExpClassPair | None:
    """Prefer native mlx_lm qwen4_exp; otherwise the mlx-vlm adapter."""
    native_classes = import_native_mlx_lm_qwen4_exp_classes()
    if native_classes is not None:
        return native_classes
    return import_mlx_vlm_backed_qwen4_exp_classes()


def import_exo_vendor_qwen4_exp_module() -> ModuleType | None:
    return _import_optional_module(EXO_VENDOR_QWEN4_EXP_MODULE)


def install_qwen4_exp_shim_into_mlx_lm(
    *,
    module_registry: MutableMapping[str, ModuleType] | None = None,
    import_mlx_lm_module: ModuleImporter | None = None,
    import_shim_module: ModuleImporter | None = None,
) -> Qwen4ExpImplementationSource | None:
    """Register the EXO shim as ``mlx_lm.models.qwen4_exp`` when needed.

    Side effect is scoped to ``module_registry`` (defaults to ``sys.modules``)
    so tests can install into an isolated mapping. Native mlx_lm is never
    overwritten. Handled by ``load_mlx_lm_model`` before construct.
    """
    registry = sys.modules if module_registry is None else module_registry
    load_native = (
        import_mlx_lm_module
        if import_mlx_lm_module is not None
        else import_native_mlx_lm_qwen4_exp_module
    )
    load_shim = (
        import_shim_module
        if import_shim_module is not None
        else import_exo_vendor_qwen4_exp_module
    )

    native_module = load_native()
    if native_module is not None:
        return "mlx_lm"

    shim_module = load_shim()
    if shim_module is None:
        return None
    registry[MLX_LM_QWEN4_EXP_MODULE] = shim_module
    return "mlx_vlm"
