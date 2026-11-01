"""Offline interface checks; no models, credentials, or network required."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

try:
    from .prompt_generation import generate
    from .common import (DomainDefinition, Example, PendingConfiguration,
                         PromptSpec, PromptStep, render)
    from .models import MODELS, provider_payload, render_local_chat
except ImportError:
    from prompt_generation import generate
    from common import (DomainDefinition, Example, PendingConfiguration,
                        PromptSpec, PromptStep, render)
    from models import MODELS, provider_payload, render_local_chat


class FakeBackend:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.requests = []

    def complete(self, model, request):
        self.requests.append((model, request))
        return next(self.outputs)


class GenerationTests(unittest.TestCase):
    def setUp(self):
        self.domain = DomainDefinition("test", "Test topic", "A supplied topic definition.")
        self.prompt = PromptSpec("Shared rules", (Example("Example request", "Example answer"),),
                                 (PromptStep("write", "Task for {{domain_name}}"),))

    def test_all_five_modules_are_dispatched(self):
        for method_id in ("A0", "A1", "A2", "A3", "A4"):
            prompt = self.prompt if method_id == "A1" else PromptSpec(steps=self.prompt.steps)
            backend = FakeBackend(["word " * 100])
            result = generate.generate_one(method_id, "llama31_8b", self.domain, backend, prompt)
            self.assertEqual(result.method_id, method_id)
            self.assertTrue(result.length_valid)
            self.assertEqual(result.word_count, 100)

    def test_no_default_method_runs_without_its_prompt(self):
        for method_id in generate.METHODS:
            backend = FakeBackend([])
            with self.assertRaises(PendingConfiguration):
                generate.generate_one(method_id, "llama31_8b", self.domain, backend)
            self.assertEqual(backend.requests, [])

    def test_few_shot_roles_and_provider_mapping(self):
        request = generate.METHODS["A1"].build_request(self.domain, self.prompt)
        self.assertEqual([m["role"] for m in request.messages],
                         ["system", "user", "assistant", "user"])
        original = copy.deepcopy(request.messages)
        for model in MODELS.values():
            payload = provider_payload(model, request)
            if model.provider in ("openai", "huggingface"):
                self.assertEqual(payload["messages"], request.messages)
            elif model.provider == "anthropic":
                self.assertEqual(payload["system"], "Shared rules")
                self.assertEqual(payload["messages"][1]["role"], "assistant")
                self.assertNotIn("temperature", payload)
            elif model.provider == "bedrock":
                self.assertEqual(payload["system"], [{"text": "Shared rules"}])
                self.assertEqual(payload["messages"][1]["role"], "assistant")
            else:
                self.assertEqual(payload["systemInstruction"]["parts"][0]["text"], "Shared rules")
                self.assertEqual([x["role"] for x in payload["contents"]],
                                 ["user", "model", "user"])
        self.assertEqual(request.messages, original)

    def test_multistep_passes_actual_assistant_output(self):
        prompt = PromptSpec(steps=(PromptStep("first", "First task"),
                                   PromptStep("final", "Next task: {{previous_output}}")))
        backend = FakeBackend(["Actual intermediate output", "word " * 80])
        result = generate.generate_one("A2", "qwen25_7b", self.domain, backend, prompt)
        first = backend.requests[0][1]
        last = backend.requests[1][1]
        self.assertFalse(first.is_final)
        self.assertNotIn("80–150", first.messages[-1]["content"])
        self.assertEqual(last.messages[1],
                         {"role": "assistant", "content": "Actual intermediate output"})
        self.assertIn("Next task: Actual intermediate output", last.messages[-1]["content"])
        self.assertIn("80–150", last.messages[-1]["content"])
        self.assertTrue(result.length_valid)

    def test_length_boundaries_preserve_original_output(self):
        prompt = PromptSpec(steps=self.prompt.steps)
        for count in (79, 80, 150, 151):
            text = "word " * count
            result = generate.generate_one("A4", "mistral_7b", self.domain,
                                           FakeBackend([text]), prompt)
            self.assertEqual(result.text, text)
            self.assertEqual(result.length_valid, 80 <= count <= 150)

    def test_zero_and_few_shot_require_distinct_examples(self):
        with self.assertRaises(PendingConfiguration):
            generate.METHODS["A0"].build_request(self.domain, self.prompt)
        with self.assertRaises(PendingConfiguration):
            generate.METHODS["A1"].build_request(self.domain, PromptSpec(steps=self.prompt.steps))

    def test_local_models_delegate_special_tokens_to_tokenizer(self):
        class Tokenizer:
            chat_template = "checkpoint template"

            def apply_chat_template(inner, messages, **kwargs):
                inner.received = (messages, kwargs)
                return "rendered by tokenizer"

        request = generate.METHODS["A1"].build_request(self.domain, self.prompt)
        for key in ("llama31_8b", "qwen25_7b", "mistral_7b"):
            tokenizer = Tokenizer()
            self.assertEqual(render_local_chat(MODELS[key], request, tokenizer),
                             "rendered by tokenizer")
            self.assertEqual(tokenizer.received[0], request.messages)
            self.assertEqual(tokenizer.received[1],
                             {"tokenize": False, "add_generation_prompt": True})

    def test_variables_preserve_json_and_do_not_reexpand_inserted_text(self):
        self.assertEqual(render('{"value": "{{domain_name}}"}',
                                {"domain_name": "{{literal}}"}), '{"value": "{{literal}}"}')
        with self.assertRaises(PendingConfiguration):
            render("{{unknown}}", {})

    def test_lloom_style_domain_interface_and_no_fixed_counts(self):
        domain = generate.load_domains([{"id": "a", "name": "Topic", "prompt": "Criterion"}])[0]
        self.assertEqual(domain.definition, "Criterion")
        config = generate.config_template()
        summary = generate.describe(config, generate.validate_config(config))
        self.assertIsNone(summary["domain_count"])
        self.assertIsNone(summary["planned_email_count"])
        config["domains"] = [dict(id="a", name="Topic", definition="Criterion")]
        config["samples_per_domain_per_method_per_model"] = 2
        self.assertEqual(generate.describe(config, generate.validate_config(config))[
            "planned_email_count"], 70)

    def test_cli_init_preview_and_prevent_config_overwrite(self):
        script = str(Path(generate.__file__).resolve())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            def cli(*args):
                return subprocess.run([sys.executable, script, *map(str, args)],
                                      capture_output=True, text=True)
            self.assertEqual(cli("--init-config", path).returncode, 0)
            initial = path.read_text()
            self.assertNotEqual(cli("--init-config", path).returncode, 0)
            self.assertEqual(path.read_text(), initial)
            self.assertNotEqual(cli("--config", path, "--preview").returncode, 0)
            config = json.loads(initial)
            config["domains"] = [dict(id="test", name="Topic", definition="Criterion")]
            config["prompts"]["A0"]["steps"] = [dict(name="write", user_prompt="Task")]
            path.write_text(json.dumps(config))
            preview = cli("--config", path, "--preview", "--method", "A0")
            self.assertEqual(preview.returncode, 0, preview.stderr)
            self.assertEqual(len(json.loads(preview.stdout)), 7)


if __name__ == "__main__":
    unittest.main()
