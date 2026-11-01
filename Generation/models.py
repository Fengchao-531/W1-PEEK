"""Seven model entries and message-format adapters; no inference calls yet.

References checked 2026-10-24:
https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
https://qwen.readthedocs.io/en/v2.5/inference/chat.html
https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3?chat_template=default
https://developers.openai.com/api/docs/models/gpt-4o-mini
https://docs.aws.amazon.com/nova/latest/userguide/code-examples-converse.html
https://platform.claude.com/docs/en/models/sonnet-5-5/overview
https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash

Mistral's v0.3 Instruct checkpoint is provisional. Exact checkpoint revisions,
API snapshots, AWS region/inference profile, and all decoding settings are TODO.
"""
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

try:
    from .common import GenerationRequest
except ImportError:
    from common import GenerationRequest


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    provider: str
    model_id: str
    revision: Optional[str] = None


MODELS = {
    "llama31_8b": ModelSpec("llama31_8b", "Llama 3.1 8B Instruct", "huggingface",
                            "meta-llama/Llama-3.1-8B-Instruct"),
    "qwen25_7b": ModelSpec("qwen25_7b", "Qwen 2.5 7B Instruct", "huggingface",
                           "Qwen/Qwen2.5-7B-Instruct"),
    "mistral_7b": ModelSpec("mistral_7b", "Mistral 7B Instruct v0.3", "huggingface",
                           "mistralai/Mistral-7B-Instruct-v0.3"),
    "gpt4o_mini": ModelSpec("gpt4o_mini", "GPT-4o mini", "openai", "gpt-4o-mini"),
    "nova_micro": ModelSpec("nova_micro", "Amazon Nova Micro 1.0", "bedrock",
                           "amazon.nova-micro-v1:0"),
    "claude_sonnet55": ModelSpec("claude_sonnet55", "Claude Sonnet 5.5", "anthropic",
                                "claude-sonnet-5-5"),
    "gemini38_flash": ModelSpec("gemini38_flash", "Gemini 3.8 Flash", "google",
                               "gemini-3.8-flash"),
}


def split_messages(messages: List[Dict[str, str]]):
    """One optional initial system turn, then alternating user/assistant turns."""
    if not messages:
        raise ValueError("Messages must end in a user request")
    system = ""
    turns = [dict(message) for message in messages]
    if turns[0].get("role") == "system":
        system = turns.pop(0)["content"]
    if not turns or len(turns) % 2 != 1:
        raise ValueError("Conversation must start/end with user and alternate roles")
    for index, turn in enumerate(turns):
        if turn.get("role") != ("user" if index % 2 == 0 else "assistant"):
            raise ValueError("Expected alternating user/assistant turns")
        if not isinstance(turn.get("content"), str) or not turn["content"].strip():
            raise ValueError("Message content must be nonempty text")
    if not isinstance(system, str):
        raise ValueError("System content must be text")
    return system, turns


def render_local_chat(model: ModelSpec, request: GenerationRequest, tokenizer) -> str:
    """Use the actual checkpoint tokenizer, never hand-write special tokens.

    Llama/Qwen keep system/user/assistant roles. The selected Mistral v0.3
    tokenizer accepts an initial system turn and encodes it using its own rules.
    An incompatible checkpoint/template fails explicitly; roles are not silently
    merged into user content. Return text is for future inference adapters; when
    tokenizing it later, use add_special_tokens=False to avoid duplicate BOS.
    """
    if model.provider != "huggingface":
        raise ValueError("render_local_chat requires a Hugging Face model")
    split_messages(request.messages)
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("Tokenizer has no chat template; verify checkpoint/version")
    return tokenizer.apply_chat_template(
        request.messages, tokenize=False, add_generation_prompt=True)


def provider_payload(model: ModelSpec, request: GenerationRequest) -> dict:
    """Message routing only; token/sampling/authentication fields are pending.

    Google uses the REST generateContent field names. Claude and Bedrock keep
    system outside messages. No proprietary adapter requires literal role tokens.
    """
    system, turns = split_messages(request.messages)
    if model.provider in ("openai", "huggingface"):
        return {"model": model.model_id, "messages": [dict(x) for x in request.messages]}
    if model.provider == "anthropic":
        result = {"model": model.model_id, "messages": turns}
        if system:
            result["system"] = system
        # TODO: Sonnet 5.5 rejects temperature/top_p/top_k. Its thinking/effort
        # setting must be agreed before implementing the inference adapter.
        return result
    if model.provider == "bedrock":
        result = {"modelId": model.model_id, "messages": [
            {"role": x["role"], "content": [{"text": x["content"]}]} for x in turns]}
        if system:
            result["system"] = [{"text": system}]
        return result
    if model.provider == "google":
        result = {"model": model.model_id, "contents": [
            {"role": "model" if x["role"] == "assistant" else "user",
             "parts": [{"text": x["content"]}]} for x in turns]}
        if system:
            result["systemInstruction"] = {"parts": [{"text": system}]}
        return result
    raise ValueError("Unknown provider: " + model.provider)


def model_catalog() -> dict:
    return {key: asdict(value) for key, value in MODELS.items()}
