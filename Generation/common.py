"""Shared interfaces; no paper-specific prompts or experimental settings here.

Word length always means 80--150 words in the complete returned email, including
the subject if present. Alphanumeric runs count as words; internal apostrophes
and hyphens stay in the same word. Token limits are a separate future setting.
"""
from dataclasses import dataclass, field
import re
from typing import Dict, List, Optional, Protocol, Tuple

MIN_WORDS, MAX_WORDS = 80, 150
WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:['’\-][A-Za-z0-9]+)*")
VARIABLE_PATTERN = re.compile(r"\{\{\s*([a-z_]+)\s*\}\}")


class PendingConfiguration(ValueError):
    """The user has not supplied a domain, prompt, or method procedure yet."""


@dataclass(frozen=True)
class DomainDefinition:
    """Topic definition, analogous to LLooM's name/prompt, not a DNS domain.

    Supply the chosen annotation concept's name and criterion as name/definition.
    No existing annotation domains are selected or imported automatically.
    """
    id: str
    name: str
    definition: str

    def __post_init__(self):
        for key in ("id", "name", "definition"):
            value = getattr(self, key)
            if not isinstance(value, str) or not value.strip():
                raise PendingConfiguration("Domain %s must be a nonempty string" % key)


@dataclass(frozen=True)
class Example:
    user: str
    assistant: str


@dataclass(frozen=True)
class PromptStep:
    name: str
    user_prompt: str


@dataclass(frozen=True)
class PromptSpec:
    """All steps use the selected model; the last step produces the email.

    A2/A3 step count, wording, and procedure remain unspecified. This interface
    supports sequential steps but does not claim to reproduce either paper.
    A3 optimizer/search/evaluation logic, if required, must be implemented later.
    """
    system_prompt: str = ""
    examples: Tuple[Example, ...] = ()
    steps: Tuple[PromptStep, ...] = ()


@dataclass(frozen=True)
class GenerationRequest:
    method_id: str
    domain_id: str
    step_name: str
    is_final: bool
    messages: List[Dict[str, str]]


@dataclass(frozen=True)
class GenerationResult:
    method_id: str
    domain_id: str
    text: str
    word_count: int
    length_valid: bool
    step_outputs: Tuple[str, ...] = field(default_factory=tuple)


class Backend(Protocol):
    """Future adapter: make one model call and return assistant text only.

    The adapter must reject truncated, refused, or blocked responses rather than
    return them as completed emails. API credentials and decoding settings belong
    to the adapter, not to prompt-method modules.
    """
    def complete(self, model, request: GenerationRequest) -> str:
        ...


def prompt_from_dict(data: dict) -> PromptSpec:
    if not isinstance(data, dict):
        raise PendingConfiguration("Each method prompt must be an object")
    return PromptSpec(
        system_prompt=data.get("system_prompt") or "",
        examples=tuple(Example(**x) for x in data.get("examples", [])),
        steps=tuple(PromptStep(**x) for x in data.get("steps", [])),
    )


def validate_prompt(method_id: str, prompt: PromptSpec):
    if not prompt.steps:
        raise PendingConfiguration("%s: prompt and procedure are pending" % method_id)
    if not isinstance(prompt.system_prompt, str):
        raise PendingConfiguration("system_prompt must be a string")
    names = set()
    for step in prompt.steps:
        if not isinstance(step.name, str) or not step.name.strip() or step.name in names:
            raise PendingConfiguration("Step names must be nonempty and unique")
        names.add(step.name)
        if not isinstance(step.user_prompt, str) or not step.user_prompt.strip():
            raise PendingConfiguration("%s/%s: user_prompt is pending" % (method_id, step.name))
    if method_id == "A0" and prompt.examples:
        raise PendingConfiguration("A0 Zero-shot must not contain few-shot examples")
    if method_id == "A1" and not prompt.examples:
        raise PendingConfiguration("A1 Few-shot examples are pending")
    for example in prompt.examples:
        if any(not isinstance(x, str) or not x.strip()
               for x in (example.user, example.assistant)):
            raise PendingConfiguration("Each example needs user and assistant text")


def render(text: str, values: dict) -> str:
    """Replace named {{variables}} once; leave ordinary JSON braces untouched."""
    def replace(match):
        key = match.group(1)
        if key not in values:
            raise PendingConfiguration("Unknown or unavailable prompt variable: " + key)
        return str(values[key])
    return VARIABLE_PATTERN.sub(replace, text)


def build_request(method_id: str, domain: DomainDefinition, prompt: PromptSpec,
                  step_index: int = 0, previous_outputs: Tuple[str, ...] = ()) -> GenerationRequest:
    validate_prompt(method_id, prompt)
    if not 0 <= step_index < len(prompt.steps) or len(previous_outputs) != step_index:
        raise ValueError("Provide exactly one actual assistant output per previous step")
    values = {"domain_id": domain.id, "domain_name": domain.name,
              "domain_definition": domain.definition,
              "min_words": MIN_WORDS, "max_words": MAX_WORDS}
    messages = []
    # Method instructions live in user turns. Optional common persona/rules live
    # in system; demonstrated and actual model responses live in assistant turns.
    if prompt.system_prompt.strip():
        messages.append({"role": "system", "content": render(prompt.system_prompt, values)})
    for example in prompt.examples:
        messages.extend([
            {"role": "user", "content": render(example.user, values)},
            {"role": "assistant", "content": render(example.assistant, values)},
        ])
    for index, step in enumerate(prompt.steps[:step_index + 1]):
        step_values = dict(values)
        if index:
            step_values["previous_output"] = previous_outputs[index - 1]
        content = render(step.user_prompt, step_values)
        # Shared experimental context is added consistently across all methods.
        content += "\n\nTopic domain: %s\nDomain definition (reference):\n%s" % (
            domain.name, domain.definition)
        if index == len(prompt.steps) - 1:
            content += ("\n\nReturn only one complete email in English, with no commentary. "
                        "Use 80–150 words in total, including the subject if present.")
        messages.append({"role": "user", "content": content})
        if index < step_index:
            output = previous_outputs[index]
            if not isinstance(output, str) or not output.strip():
                raise ValueError("Previous step output must be nonempty assistant text")
            messages.append({"role": "assistant", "content": output})
    return GenerationRequest(method_id, domain.id, prompt.steps[step_index].name,
                             step_index == len(prompt.steps) - 1, messages)


def execute(method_id: str, domain: DomainDefinition, model, backend: Backend,
            prompt: PromptSpec) -> GenerationResult:
    """Run configured steps; flag length failures without rewriting or truncating.

    Sampling, retries, persistence and dataset scheduling are deferred. No SFT,
    critique, discriminator, adversarial training, or ablation is performed.
    """
    validate_prompt(method_id, prompt)
    outputs = []
    for index in range(len(prompt.steps)):
        request = build_request(method_id, domain, prompt, index, tuple(outputs))
        output = backend.complete(model, request)
        if not isinstance(output, str) or not output.strip():
            raise ValueError("Backend returned empty or non-text assistant output")
        outputs.append(output)
    text = outputs[-1]
    count = len(WORD_PATTERN.findall(text))
    return GenerationResult(method_id, domain.id, text, count,
                            MIN_WORDS <= count <= MAX_WORDS, tuple(outputs))
