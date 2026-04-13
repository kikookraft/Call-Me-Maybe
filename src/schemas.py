from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class ParameterSpec(BaseModel):
    """Schema for a single parameter definition."""

    model_config: dict[str, Any] = ConfigDict(extra="forbid")

    type: str


class ReturnSpec(BaseModel):
    """Schema for a function return definition."""

    model_config: dict[str, Any] = ConfigDict(extra="forbid")

    type: str


class FunctionDefinition(BaseModel):
    """Schema for one available function."""

    model_config: dict[str, Any] = ConfigDict(extra="forbid")

    name: str
    description: str
    parameters: dict[str, ParameterSpec]
    returns: ReturnSpec


class InputPrompt(BaseModel):
    """Schema for one input prompt."""

    model_config: dict[str, Any] = ConfigDict(extra="forbid")

    prompt: str


class FunctionCallResult(BaseModel):
    """Schema for the final output object."""

    model_config: dict[str, Any] = ConfigDict(extra="forbid")

    prompt: str
    name: str
    parameters: dict[str, Any]
