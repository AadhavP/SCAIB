"""Built-in provider/runtime adapters with lazy SDK boundaries."""

from agent_evals.agents.backends.anthropic import AnthropicRuntime
from agent_evals.agents.backends.custom import CustomPythonRuntime
from agent_evals.agents.backends.external_process import ExternalProcessRuntime
from agent_evals.agents.backends.openai import OpenAIRuntime
from agent_evals.agents.backends.openai_compatible import OpenAICompatibleRuntime

__all__ = [
    "AnthropicRuntime",
    "CustomPythonRuntime",
    "ExternalProcessRuntime",
    "OpenAICompatibleRuntime",
    "OpenAIRuntime",
]
