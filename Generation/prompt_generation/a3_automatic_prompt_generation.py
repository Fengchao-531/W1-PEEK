"""A3: Automatic Prompt Generation.

TODO: Prompt 生成/搜索/评估算法待文章确认；优化器、数据划分和最终生成 Prompt 均未实现。
This is an interface, not a reproduction of the linked paper.
"""
try:
    from ..common import PromptSpec, build_request as _build_request, execute
except ImportError:
    from common import PromptSpec, build_request as _build_request, execute

METHOD_ID = "A3"
METHOD_NAME = "Automatic Prompt Generation"
# Empty until the user supplies the paper/method settings. No invented baseline.
PROMPT = PromptSpec()


def build_request(domain, prompt=None, step_index=0, previous_outputs=()):
    return _build_request(METHOD_ID, domain, PROMPT if prompt is None else prompt,
                          step_index, tuple(previous_outputs))


def generate(domain, model, backend, prompt=None):
    """Called by generate.py; backend and actual prompt are injected explicitly."""
    return execute(METHOD_ID, domain, model, backend,
                   PROMPT if prompt is None else prompt)
