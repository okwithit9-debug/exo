from exo.shared.types.common import ModelId


def get_eos_token_ids_for_model(model_id: ModelId) -> list[int] | None:
    """
    Get the EOS token IDs for a model based on its ID.

    Some models require explicit EOS token configuration that isn't in their
    tokenizer config. This function returns the known EOS token IDs for such models.

    Args:
        model_id: The HuggingFace model ID

    Returns:
        List of EOS token IDs, or None if the model uses standard tokenizer config
    """
    model_id_lower = model_id.lower()
    if "kimi-k2" in model_id_lower:
        return [163586]
    if "glm-5" in model_id_lower:
        # 154820: <|endoftext|>, 154827: <|user|>, 154829: <|observation|>
        return [154820, 154827, 154829]
    if "glm" in model_id_lower:
        # For GLM-4.7 and older
        return [151336, 151329, 151338]
    if "gpt-oss" in model_id_lower:
        return [200002, 200012]
    if (
        "qwen3.5" in model_id_lower
        or "qwen-3.5" in model_id_lower
        or "qwen3.6" in model_id_lower
        or "qwen-3.6" in model_id_lower
        or "qwen3.8" in model_id_lower
        or "qwen-3.8" in model_id_lower
    ):
        # Qwen3.5 / 3.6 / 3.8 (including Flash-Next): 248046 (<|im_end|>), 248044 (<|endoftext|>)
        return [248046, 248044]
    if "gemma-4" in model_id_lower or "gemma-3" in model_id_lower:
        return [1, 106, 50]
    return None
