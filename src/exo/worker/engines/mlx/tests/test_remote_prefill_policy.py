from typing import cast
from unittest.mock import patch

import mlx.core as mx
import pytest
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.types.common import ModelId
from exo.worker.engines.mlx.cache import _detached_copy
from exo.worker.engines.mlx.generator.generate import (
    RemotePrefillRequired,
    environment_flag_enabled,
    remote_prefill_minimum_tokens,
    resolve_prefill_mode,
    token_ids_from_prompt,
    warmup_inference,
)
from exo.worker.engines.mlx.types import Model


def test_environment_flag_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EXO_REQUIRE_REMOTE_PREFILL", raising=False)
    assert environment_flag_enabled("EXO_REQUIRE_REMOTE_PREFILL") is False
    monkeypatch.setenv("EXO_REQUIRE_REMOTE_PREFILL", "1")
    assert environment_flag_enabled("EXO_REQUIRE_REMOTE_PREFILL") is True
    monkeypatch.setenv("EXO_REQUIRE_REMOTE_PREFILL", "true")
    assert environment_flag_enabled("EXO_REQUIRE_REMOTE_PREFILL") is True
    monkeypatch.setenv("EXO_REQUIRE_REMOTE_PREFILL", "no")
    assert environment_flag_enabled("EXO_REQUIRE_REMOTE_PREFILL") is False


def test_remote_prefill_minimum_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EXO_REMOTE_PREFILL_MIN_TOKENS", raising=False)
    assert remote_prefill_minimum_tokens() == 1000
    monkeypatch.setenv("EXO_REMOTE_PREFILL_MIN_TOKENS", "1")
    assert remote_prefill_minimum_tokens() == 1


def test_resolve_prefill_mode_default_threshold() -> None:
    assert (
        resolve_prefill_mode(
            500,
            "10.10.10.2:9",
            require_remote=False,
            minimum_tokens=1000,
        )
        == "local"
    )
    assert (
        resolve_prefill_mode(
            1001,
            "10.10.10.2:9",
            require_remote=False,
            minimum_tokens=1000,
        )
        == "remote"
    )
    assert (
        resolve_prefill_mode(
            1001,
            None,
            require_remote=False,
            minimum_tokens=1000,
        )
        == "local"
    )


def test_resolve_prefill_mode_requires_endpoint() -> None:
    with pytest.raises(RemotePrefillRequired, match="prefill_endpoint is missing"):
        resolve_prefill_mode(
            2,
            None,
            require_remote=True,
            minimum_tokens=1000,
        )
    assert (
        resolve_prefill_mode(
            2,
            "10.10.10.2:9",
            require_remote=True,
            minimum_tokens=1000,
        )
        == "remote"
    )


def test_token_ids_from_prompt_coerces_array() -> None:
    tokens = mx.array([11, 22, 33])
    assert token_ids_from_prompt(tokens) == [11, 22, 33]
    assert token_ids_from_prompt([7, 8]) == [7, 8]


def test_detached_copy_evals_before_numpy_roundtrip() -> None:
    source = mx.array([1.0, 2.0, 3.0])
    with (
        patch("exo.worker.engines.mlx.cache.mx.eval") as eval_mock,
        patch("exo.worker.engines.mlx.cache.mx.synchronize") as synchronize_mock,
    ):
        copied = _detached_copy(source)
        eval_mock.assert_called_once()
        synchronize_mock.assert_called_once()
    assert copied.tolist() == [1.0, 2.0, 3.0]


def test_warmup_inference_skips_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXO_SKIP_WARMUP", "1")
    check_every = warmup_inference(
        model=cast(Model, object()),
        tokenizer=cast(TokenizerWrapper, object()),
        group=None,
        model_id=ModelId("test-model"),
    )
    assert check_every == 50
