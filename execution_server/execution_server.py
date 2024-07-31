from asyncio.subprocess import Process
import math
import threading
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import subprocess
import tempfile
import traceback
import typer
import os
from dataclasses import dataclass
import logging

import concurrent

# NOTE: Schema is moved into place by the dockerfile
from schema import ExecuteCodeRequest, ExecuteCodeResult

app = FastAPI()

import time


@app.get("/ping")
async def ping():
    return "pong"


MAX_TIMEOUT = 65

# Per worker
MAX_CONCURRENT_PROGRAMS = 32

REQUEST_TIMEOUT_SECONDS = 3600

TEMPFILE_DIR = os.environ.get("TEMPFILE_DIR", None)

semaphore = threading.Semaphore(MAX_CONCURRENT_PROGRAMS)


def execute_with_input(
    code_file_name: str, input_str: str, timeout: float, memory_limit_bytes: int
):
    with semaphore:
        with tempfile.NamedTemporaryFile(mode="w+", delete=True) as temp_input_file:
            temp_input_file.write(input_str)
            temp_input_file.flush()  # Ensure the data is written to disk

            # Rewind the file so the subprocess reads it from the beginning
            temp_input_file.seek(0)

            try:
                memory_limit_kb = math.ceil(memory_limit_bytes / 1024)
                cmd = f"ulimit -v {memory_limit_kb} && python {code_file_name}"

                try:
                    result = subprocess.run(
                        ["bash", "-c", cmd],
                        capture_output=True,
                        stdin=temp_input_file.fileno(),
                        timeout=timeout,
                    )
                except subprocess.TimeoutExpired:

                    return ExecuteCodeResult(
                        input=input_str,
                        stdout="",
                        stderr="",
                        return_code=None,
                        timed_out=True,
                    )

                return ExecuteCodeResult(
                    input=input_str,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    return_code=result.returncode,
                    timed_out=False,
                )
            except Exception as e:
                exception_str = traceback.format_exc()
                return ExecuteCodeResult(
                    input=input_str,
                    stdout="",
                    stderr=f"Python Exception Thrown: {exception_str}",
                    return_code=None,
                    timed_out=False,
                )


@app.post("/execute", response_model=List[ExecuteCodeResult])
def execute_python_code(request: ExecuteCodeRequest):
    start = time.time()
    if request.timeout > MAX_TIMEOUT:
        raise HTTPException(
            status_code=400,
            detail=f"Timeout must be less than {MAX_TIMEOUT} seconds",
        )

    try:
        with tempfile.NamedTemporaryFile(suffix=".py", delete=True) as temp:
            with open(temp.name, "w") as f:
                f.write(request.code)

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=MAX_CONCURRENT_PROGRAMS
            ) as executor:
                futures = [
                    executor.submit(
                        execute_with_input,
                        code_file_name=temp.name,
                        input_str=input_str,
                        timeout=request.timeout,
                        memory_limit_bytes=request.memory_limit_bytes,
                    )
                    for input_str in request.inputs
                ]
                results = [future.result() for future in futures]

            return results

    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


def main(port: int = 8004):
    logging_level = os.environ.get("LOGGING_LEVEL", "WARNING")
    print(f"Setting logging level to {logging_level}")

    LOG_ENUM = getattr(logging, logging_level)
    logging.getLogger("uvicorn").setLevel(LOG_ENUM)
    logging.getLogger("uvicorn.access").setLevel(LOG_ENUM)
    logging.getLogger("uvicorn.error").setLevel(LOG_ENUM)

    print(f"Starting server with {MAX_CONCURRENT_PROGRAMS} max concurrent programs")

    print(f"Starting server on port {port}")
    uvicorn.run(
        "execution_server:app",
        host="0.0.0.0",
        port=port,
        timeout_keep_alive=REQUEST_TIMEOUT_SECONDS,
        backlog=8192 * 2,
        workers=4,
    )


if __name__ == "__main__":
    typer.run(main)
