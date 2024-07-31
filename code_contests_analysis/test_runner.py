from tqdm import tqdm
from .cc_dataset import (
    Problem,
    Solution,
    TestCaseRun,
    DEFAULT_TIMEOUT_SECONDS,
    DatasetSplit,
)
import uuid

import concurrent.futures
from .execution_server_client import ExecutionServerClient

MAX_WORKERS = 128


def run_tests_on_problem_and_solution(
    problem: Problem, solution: Solution, execution_server: ExecutionServerClient
) -> TestCaseRun:
    inputs = [test_case.input for test_case in problem.test_cases]

    results = execution_server.execute_code(
        solution.code,
        inputs,
        timeout=DEFAULT_TIMEOUT_SECONDS,
        memory_limit_bytes=4 * problem.memory_limit_bytes,  # give a wide buffer
    )

    if len(results) != len(inputs):
        raise ValueError(
            "Execution server returned incorrect number of results."
            f" Expected {len(inputs)}, got {len(results)}"
        )

    test_case_runs = []
    for test_case, result in zip(problem.test_cases, results):
        if test_case.input != result.input:
            raise ValueError(
                "Unexpected input mismatch."
                f" we sent `{test_case.input}` to the server, server returned `{result.input}`"
            )

        test_case_run_id = uuid.uuid1()
        test_case_runs.append(
            TestCaseRun(
                id=test_case_run_id,
                solution=solution,
                test_case=test_case,
                stdout=result.stdout,
                stderr=result.stderr,
                return_code=result.return_code,
                timed_out=result.timed_out,
            )
        )

    return test_case_runs


def run_tests(problems: Problem, num_workers: int):
    with ExecutionServerClient() as client:
        futures = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            for problem in problems:
                if problem.dataset_split != DatasetSplit.TEST:
                    continue

                for solution in problem.solutions:
                    if solution.is_correct:
                        futures.append(
                            executor.submit(
                                run_tests_on_problem_and_solution,
                                problem=problem,
                                solution=solution,
                                execution_server=client,
                            )
                        )

            for future in tqdm(
                concurrent.futures.as_completed(futures), total=len(futures)
            ):
                yield future.result()
