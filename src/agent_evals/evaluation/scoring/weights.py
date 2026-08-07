"""Frozen benchmark weighting primitives."""

from pydantic import BaseModel, Field


class FrozenWeight(BaseModel):
    """A named weight that is part of a versioned benchmark contract."""

    name: str
    weight: float = Field(gt=0)


__all__ = ["FrozenWeight"]
