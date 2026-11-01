#!/usr/bin/env python3
"""Main entry point for A0--A4 generation interfaces. Python 3.9+, stdlib only.

From the Generation directory:
    python prompt_generation/generate.py --list-models
    python prompt_generation/generate.py --init-config generation_config.json
    python prompt_generation/generate.py --config generation_config.json
    python prompt_generation/generate.py --config generation_config.json --preview --method A0

Package invocation from W1-PEEK is also supported:
    python -m Generation.prompt_generation.generate --list-models

No model downloads or API calls occur through this CLI. Actual backend adapters,
dataset execution, seeds, sampling, persistence and retry policy are deferred.
Use generate_one(..., backend=...) after implementing a Backend.complete adapter.
"""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
from typing import Optional

if not __package__:
    # Direct script execution: expose Generation's shared modules and import the
    # five methods through their package, regardless of the current directory.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "prompt_generation"

from . import (a0_zero_shot, a1_few_shot, a2_multiple_generation,
               a3_automatic_prompt_generation, a4_peek)

try:
    from ..common import DomainDefinition, PendingConfiguration, prompt_from_dict, validate_prompt
    from ..models import MODELS, model_catalog, provider_payload
except ImportError:
    from common import DomainDefinition, PendingConfiguration, prompt_from_dict, validate_prompt
    from models import MODELS, model_catalog, provider_payload

METHODS = {module.METHOD_ID: module for module in (
    a0_zero_shot, a1_few_shot, a2_multiple_generation,
    a3_automatic_prompt_generation, a4_peek)}


def config_template():
    return {
        "schema_version": 1,
        "domains": [],
        "samples_per_domain_per_method_per_model": None,
        "models": list(MODELS),
        "model_overrides": {},
        "methods": list(METHODS),
        "prompts": {key: {"system_prompt": None, "examples": [], "steps": []}
                    for key in METHODS},
        "settings": {
            "seed": None, "temperature": None, "top_p": None,
            "max_output_tokens": None, "batch_size": None,
            "retry_policy": None, "output_directory": None,
            "aws_region": None,
        },
    }


def load_domains(definitions):
    """Public interface for future LLooM annotation concept definitions.

    Input: [{"id": "...", "name": "...", "definition": "..."}].
    The existing LLooM criterion called 'prompt' can also supply definition.
    This does not rerun LLooM, choose categories, or inherit all 16 old domains.
    """
    if not isinstance(definitions, list):
        raise PendingConfiguration("domains must be a list of domain definitions")
    domains = []
    seen = set()
    for item in definitions:
        if not isinstance(item, dict):
            raise PendingConfiguration("Each domain must be an object")
        domain = DomainDefinition(item.get("id"), item.get("name"),
                                  item.get("definition", item.get("prompt")))
        if domain.id in seen:
            raise PendingConfiguration("Duplicate domain id: " + domain.id)
        seen.add(domain.id)
        domains.append(domain)
    return domains


def resolve_model(key: str, config: Optional[dict] = None):
    model = MODELS[key]
    override = (config or {}).get("model_overrides", {}).get(key, {})
    if set(override) - {"model_id", "revision"}:
        raise PendingConfiguration("Model overrides support only model_id and revision")
    for field, value in override.items():
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise PendingConfiguration("Model override %s must be a nonempty string" % field)
    if "model_id" in override and override["model_id"] is None:
        raise PendingConfiguration("model_id cannot be null")
    return replace(model, **override)


def generate_one(method_id, model_key, domain, backend, prompt=None, config=None):
    """Dispatch to one of the five method files, using an injected backend."""
    if prompt is None and config is not None:
        prompt = prompt_from_dict(config.get("prompts", {}).get(method_id, {}))
    return METHODS[method_id].generate(domain, resolve_model(model_key, config), backend, prompt)


def validate_config(config):
    if config.get("schema_version") != 1:
        raise PendingConfiguration("Expected schema_version=1")
    for name, registry in (("models", MODELS), ("methods", METHODS)):
        keys = config.get(name)
        if not isinstance(keys, list) or not keys or any(
                not isinstance(key, str) or key not in registry for key in keys):
            raise PendingConfiguration("%s must be a nonempty list of registered IDs" % name)
        if len(set(keys)) != len(keys):
            raise PendingConfiguration("Duplicate %s IDs" % name)
    count = config.get("samples_per_domain_per_method_per_model")
    if count is not None and (type(count) is not int or count <= 0):
        raise PendingConfiguration("Sample count must be null or a positive integer")
    for key in config["models"]:
        resolve_model(key, config)
    return load_domains(config.get("domains", []))


def describe(config, domains):
    pending = []
    if not domains:
        pending.append("Domain selection, count and definitions")
    count = config.get("samples_per_domain_per_method_per_model")
    if count is None:
        pending.append("Samples per domain/method/model")
    for key in config["methods"]:
        try:
            validate_prompt(key, prompt_from_dict(config.get("prompts", {}).get(key, {})))
        except PendingConfiguration as error:
            pending.append(str(error))
    pending.extend([
        "A2/A3 paper-specific generation/optimization algorithms",
        "Inference backends, credentials, checkpoint revisions and API snapshots",
        "Sampling, seeds, token budgets, batching, retries and output persistence",
        "AWS region and Nova inference profile; Mistral checkpoint confirmation",
    ])
    return {
        "model_count": len(config["models"]),
        "methods": {key: METHODS[key].METHOD_NAME for key in config["methods"]},
        "domain_count": len(domains) if domains else None,
        "samples_per_domain_per_method_per_model": count,
        "planned_email_count": (len(config["models"]) * len(config["methods"]) *
                                len(domains) * count) if domains and count else None,
        "email_word_range": [80, 150],
        "pending": pending,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-config", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--list-models", action="store_true")
    parser.add_argument("--preview", action="store_true", help="Preview first-step messages only")
    parser.add_argument("--method", choices=list(METHODS))
    parser.add_argument("--model", choices=list(MODELS))
    parser.add_argument("--domain", help="Domain ID to preview")
    args = parser.parse_args(argv)
    try:
        if args.init_config:
            with args.init_config.open("x", encoding="utf-8") as stream:
                json.dump(config_template(), stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            print("Created " + str(args.init_config))
            return 0
        if args.list_models:
            print(json.dumps(model_catalog(), ensure_ascii=False, indent=2))
            return 0
        config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else config_template()
        if not isinstance(config, dict):
            raise PendingConfiguration("Config must be a JSON object")
        domains = validate_config(config)
        if not args.preview:
            print(json.dumps(describe(config, domains), ensure_ascii=False, indent=2))
            return 0
        selected_domains = [x for x in domains if args.domain is None or x.id == args.domain]
        if not selected_domains:
            raise PendingConfiguration("Supply at least one matching domain definition before preview")
        selected_methods = [args.method] if args.method else config["methods"]
        selected_models = [args.model] if args.model else config["models"]
        if any(key not in config["methods"] for key in selected_methods) or any(
                key not in config["models"] for key in selected_models):
            raise PendingConfiguration("Preview selections must be enabled in the config")
        previews = []
        for method_id in selected_methods:
            prompt = prompt_from_dict(config.get("prompts", {}).get(method_id, {}))
            for domain in selected_domains:
                request = METHODS[method_id].build_request(domain, prompt)
                for model_key in selected_models:
                    model = resolve_model(model_key, config)
                    previews.append({"model_key": model_key, "request": asdict(request),
                                     "provider_message_fields": provider_payload(model, request)})
        print(json.dumps(previews, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, TypeError, KeyError, OSError) as error:
        parser.exit(2, "Configuration error: %s\n" % error)


if __name__ == "__main__":
    raise SystemExit(main())
