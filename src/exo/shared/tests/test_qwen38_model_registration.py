import pytest
from anyio import Path

from exo.shared.constants import RESOURCES_DIR
from exo.shared.models.model_cards import ConfigData, ModelCard
from exo.shared.types.common import ModelId

_CARDS_DIR = Path(str(RESOURCES_DIR)) / "inference_model_cards"


def _qwen38_27b_config() -> dict[str, object]:
    return {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "image_token_id": 248056,
        "text_config": {
            "hidden_size": 5120,
            "num_hidden_layers": 64,
            "num_key_value_heads": 4,
            "max_position_embeddings": 262144,
        },
        "vision_config": {
            "model_type": "qwen3_5",
            "hidden_size": 1152,
        },
    }


def _qwen38_flash_next_config() -> dict[str, object]:
    return {
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "model_type": "qwen4_exp",
        "image_token_id": 248056,
        "text_config": {
            "hidden_size": 2560,
            "num_hidden_layers": 48,
            "num_key_value_heads": 2,
            "max_position_embeddings": 262144,
            "model_type": "qwen4_exp",
        },
        "vision_config": {
            "model_type": "qwen4_exp",
            "hidden_size": 1152,
        },
    }


@pytest.mark.parametrize(
    ("filename", "model_id", "n_layers", "hidden_size"),
    [
        (
            "mlx-community--Qwen3.8-27B-4bit.toml",
            "mlx-community/Qwen3.8-27B-4bit",
            64,
            5120,
        ),
        (
            "mlx-community--Qwen3.8-27B-8bit.toml",
            "mlx-community/Qwen3.8-27B-8bit",
            64,
            5120,
        ),
        (
            "mlx-community--Qwen3.8-Flash-Next-4bit.toml",
            "mlx-community/Qwen3.8-Flash-Next-4bit",
            48,
            2560,
        ),
    ],
)
async def test_qwen38_builtin_cards_load(
    filename: str, model_id: str, n_layers: int, hidden_size: int
) -> None:
    card = await ModelCard.load_from_path(_CARDS_DIR / filename)
    assert card.model_id == ModelId(model_id)
    assert card.n_layers == n_layers
    assert card.hidden_size == hidden_size
    assert card.supports_tensor is True
    assert card.family == "qwen"
    assert "text" in card.capabilities


async def test_qwen38_27b_card_is_qwen3_5_load_compatible() -> None:
    card = await ModelCard.load_from_path(
        _CARDS_DIR / "mlx-community--Qwen3.8-27B-4bit.toml"
    )
    config = ConfigData.model_validate(
        _qwen38_27b_config(), context={"model_id": str(card.model_id)}
    )
    assert config.supports_tensor is True
    assert config.layer_count == 64
    assert config.hidden_size == 5120
    assert config.num_key_value_heads == 4
    assert config.vision is not None
    assert config.vision.model_type == "qwen3_5"


async def test_qwen38_flash_next_arch_is_allowlisted() -> None:
    config = ConfigData.model_validate(
        _qwen38_flash_next_config(),
        context={"model_id": "mlx-community/Qwen3.8-Flash-Next-4bit"},
    )
    assert config.supports_tensor is True
    assert config.layer_count == 48
    assert config.hidden_size == 2560
    assert config.vision is not None
    assert config.vision.model_type == "qwen4_exp"


def test_uncensored_qwen38_27b_is_same_architecture() -> None:
    """Uncensored is a weight pack. EXO only needs the declared architecture."""
    local_uncensored = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": {
            "hidden_size": 5120,
            "num_hidden_layers": 64,
            "num_key_value_heads": 4,
            "max_position_embeddings": 262144,
        },
    }
    config = ConfigData.model_validate(local_uncensored)
    assert config.supports_tensor is True


def test_local_mlx_dir_uses_normalized_model_id() -> None:
    model_id = ModelId("mlx-community/Qwen3.8-27B-4bit")
    assert model_id.normalize() == "mlx-community--Qwen3.8-27B-4bit"
    uncensored = ModelId("someone/Qwen3.8-27B-Uncensored-MLX-4bit")
    assert uncensored.normalize() == "someone--Qwen3.8-27B-Uncensored-MLX-4bit"
