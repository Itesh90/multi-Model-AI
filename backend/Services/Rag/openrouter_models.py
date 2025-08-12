"""
OpenRouter free and low-cost model configurations
"""

FREE_MODELS = {
    "llama-3.1-8b": {
        "name": "meta-llama/llama-3.1-8b-instruct:free",
        "description": "Meta's Llama 3.1 8B model - Great for general tasks",
        "context_length": 128000,
        "cost": "Free",
    },
    "gpt-oss-20b": {
        "name": "openai/gpt-oss-20b:free",
        "description": "Open-source 20B class model via OpenRouter",
        "context_length": 8192,
        "cost": "Free",
    },
    "mistral-7b": {
        "name": "mistralai/mistral-7b-instruct:free",
        "description": "Mistral 7B - Fast and efficient for most tasks",
        "context_length": 32768,
        "cost": "Free",
    },
    "mythomist-7b": {
        "name": "gryphe/mythomist-7b:free",
        "description": "MythoMist 7B - Good for creative and conversational tasks",
        "context_length": 32768,
        "cost": "Free",
    },
    "starcoder": {
        "name": "huggingface/starcoder:free",
        "description": "StarCoder - Specialized for code generation",
        "context_length": 8192,
        "cost": "Free",
    },
    "openchat-3.5": {
        "name": "openchat/openchat-3.5-1210:free",
        "description": "OpenChat 3.5 - Good balance of performance and speed",
        "context_length": 8192,
        "cost": "Free",
    },
    
}

LOW_COST_MODELS = {
    "llama-3.1-70b": {
        "name": "meta-llama/llama-3.1-70b-instruct",
        "description": "Larger Llama model with better reasoning",
        "context_length": 128000,
        "cost": "$0.52/1M tokens",
    },
    "claude-haiku": {
        "name": "anthropic/claude-3-haiku",
        "description": "Fast Claude model, good for simple tasks",
        "context_length": 200000,
        "cost": "$0.25/1M tokens",
    },
}


def get_model_info(model_key: str) -> dict | None:
    """Get model information by key"""
    if model_key in FREE_MODELS:
        return FREE_MODELS[model_key]
    if model_key in LOW_COST_MODELS:
        return LOW_COST_MODELS[model_key]
    return None


def list_free_models() -> list[str]:
    """List all available free models"""
    return list(FREE_MODELS.keys())


def list_all_models() -> dict:
    """Get all available models"""
    return {**FREE_MODELS, **LOW_COST_MODELS}


