"""Shared types for interacting with the execution server.

NOTE: this file gets copied by the Dockerfile into image that runs
the execution server."""

from typing import List

from pydantic import BaseModel


class ExecuteCodeRequest(BaseModel):
    code: str
    inputs: List[str]
    timeout: float
    memory_limit_bytes: int


class ExecuteCodeResult(BaseModel):
    input: str
    stdout: str
    stderr: str
    return_code: int | None
    timed_out: bool
