"""mlx_lm.models.qwen4_exp stand-in backed by mlx-vlm.

Temporary. Delete this file when ``import mlx_lm.models.qwen4_exp`` works
on the EXO mlx-lm pin (ml-explore/mlx-lm#1788 or equivalent). The loader
hook in ``qwen4_exp_shim.py`` / ``mlx_lm_load.py`` must lose its mlx-vlm
fallback in the same change.
"""

from __future__ import annotations

from exo.worker.engines.mlx.qwen4_exp_shim import (
    import_mlx_vlm_backed_qwen4_exp_classes,
)

_loaded_classes = import_mlx_vlm_backed_qwen4_exp_classes()
if _loaded_classes is None:
    raise ImportError(
        "EXO qwen4_exp shim requires mlx-vlm>=0.6.17 "
        "(mlx_vlm.models.qwen4_exp) and mlx.nn. "
        "Install with: pip install 'mlx-vlm>=0.6.17'"
    )

Model, ModelArgs = _loaded_classes
EXO_QWEN4_EXP_SHIM: bool = True
