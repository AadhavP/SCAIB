"""Custom exceptions for agent-evals."""


class AgentEvalsError(Exception):
    """Base exception for all agent-evals errors."""

    pass


class ConfigurationError(AgentEvalsError):
    """Raised when configuration is invalid or missing."""

    pass


class RegistryError(AgentEvalsError):
    """Raised when registering or resolving benchmarks/agents fails."""

    pass


class BenchmarkExecutionError(AgentEvalsError):
    """Raised when benchmark execution fails."""

    pass


class SandboxExecutionError(AgentEvalsError):
    """Raised when execution inside sandbox fails."""

    pass


class EnvironmentStateError(AgentEvalsError):
    """Raised when an environment or episode lifecycle transition is invalid."""

    pass


class DatasetNotFoundError(AgentEvalsError):
    """Raised when requested dataset cannot be found or loaded."""

    pass


class EvaluationError(AgentEvalsError):
    """Raised when evaluator fails to calculate metrics."""

    pass
