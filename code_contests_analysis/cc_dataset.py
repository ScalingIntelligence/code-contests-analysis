"""Datastructures & functions for interacting with the code contests dataset.

See https://huggingface.co/datasets/deepmind/code_contests for details about
the data structure used in the dataset.
"""

from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
from typing import Any, Dict, List
from functools import cached_property, partial

from tqdm import tqdm

from datasets import load_dataset
from code_contests_analysis import compare_results


# These constants are used by other files.
DEFAULT_TIMEOUT_SECONDS = 60.0


class TestCaseSource(Enum):
    """The source of a test case."""

    PUBLIC = "public"
    PRIVATE = "private"
    GENERATED = "generated"


class DatasetSplit(Enum):
    VALID = "valid"
    TEST = "test"


@dataclass
class TestCase:
    """A test case for a specific problem.

    Attrs:
        id: a unique id (assigned by us).
        problem: the problem this test case is for.
        input: the input to the program (comes from
            example['{generated, public, private}_tests']['output'])
        expected_output: the expected output (comes from
            example['{generated, public, private}_tests']['output']).
        source: if the test case is public/private/generated.
    """

    id: str

    problem: "Problem"
    input: str
    expected_output: str
    source: TestCaseSource

    def __repr__(self):
        return f"TestCase(id='{self.id}', problem_id='{self.problem.id}', source={self.source})"


@dataclass
class Solution:
    """A solution to a problem (in python3).

    Attrs:
        id: a unique id (assigned by us).
        problem: the problem this solution case is for.
        is_correct: True if the solution was in the "solutions" field of an example, False if it's
            in the "incorrect_solutions" field of an example.
        code: the source code.
        test_case_runs: the result of running this solution on a test case. This field
            is populated by us and is not from the huggingface dataset.
    """

    id: str

    problem: "Problem"
    is_correct: bool

    code: str

    test_case_runs: List["TestCaseRun"] = field(default_factory=list)

    def __repr__(self):
        return f"Solution(id='{self.id}', problem_id='{self.problem.id}', is_correct={self.is_correct})"


@dataclass
class TestCaseRun:
    """The result of a test case on a solution.

    This data is from us running tests, not from the huggingface dataset.

    Attrs:
        id: a unique id (assigned by us).
        solution: the solution this test case was run on.
        test_case: the test case the solution was ran on.
        stdout: the stdout captured when the test case was ran.
        stderr: the stderr captured when the test case was ran.
        return_code: the return code. None if there was an unexpected error or it timed out.
        timed_out: indicates if the test timed out.
    """

    id: str

    solution: Solution
    test_case: TestCase

    stdout: str
    stderr: str
    return_code: int | None
    timed_out: bool

    @cached_property
    def is_correct(self):
        if self.return_code != 0:
            return False

        if self.timed_out:
            return False

        return compare_results.outputs_match(
            self.stdout, self.test_case.expected_output
        )

    @property
    def crashed(self):
        return self.return_code != 0

    def to_dict(self) -> dict:
        return {
            "id": str(self.id),
            "solution_id": self.solution.id,
            "test_case_id": self.test_case.id,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "return_code": self.return_code,
            "timed_out": self.timed_out,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict,
        solution_id_to_solution: Dict[str, Solution],
        test_case_id_to_test_case: Dict[str, TestCase],
    ):
        assert data.keys() == {
            "id",
            "solution_id",
            "test_case_id",
            "stdout",
            "stderr",
            "return_code",
            "timed_out",
        }

        solution_id = data["solution_id"]
        if solution_id not in solution_id_to_solution:
            raise ValueError(f"Invalid solution id {solution_id}")

        test_case_id = data["test_case_id"]
        if test_case_id not in test_case_id_to_test_case:
            raise ValueError(f"Invalid test case id {test_case_id}")

        solution = solution_id_to_solution[solution_id]
        test_case = test_case_id_to_test_case[test_case_id]

        if solution.problem != test_case.problem:
            raise ValueError(
                f"Invalid data: Solution is for problem {solution.problem.id},"
                f" but test case is for problem {test_case.problem.id} "
            )

        result = cls(
            id=data["id"],
            stdout=data["stdout"],
            stderr=data["stderr"],
            return_code=data["return_code"],
            timed_out=data["timed_out"],
            solution=solution,
            test_case=test_case,
        )
        solution.test_case_runs.append(result)
        return result

    def __repr__(self):
        return f"TestCaseRun(id='{self.id}', solution_id='{self.solution.id}', test_case_id='{self.test_case.id}', timed_out={self.timed_out})"


@dataclass
class TimeLimit:
    """Data from the 'time_limit' field of an example."""

    seconds: int
    nanos: int

    @property
    def total_seconds(self) -> float:
        return self.seconds + 10e-9 * self.nanos


@dataclass
class Problem:
    """A code contests problem.

    Attrs:
        id (str): Unique identifier for the problem.
        name (str): Name of the problem, from the 'name' field
        solutions (List[Solution]): List of solutions for this problem.
        test_cases (List[TestCase]): List of test cases for this problem.
        dataset_split (DatasetSplit): The dataset split this problem belongs to.
        time_limit (TimeLimit | None): Time limit for the problem.
        memory_limit_bytes (int): Memory limit in bytes.
        input_file (str): Input file name.
        output_file (str): Output file name.
    """

    id: str
    name: str

    solutions: List[Solution]
    test_cases: List[TestCase]

    dataset_split: DatasetSplit

    time_limit: TimeLimit | None
    memory_limit_bytes: int

    input_file: str
    output_file: str

    def __repr__(self):
        return f"Problem(id='{self.id}', name='{self.name}', dataset_split={self.dataset_split}, solution_count={len(self.solutions)}, test_case_count={len(self.test_cases)})"


# https://huggingface.co/datasets/deepmind/code_contests#data-fields
PYTHON_3_LANG = 3


def _load_solutions(
    solutions_data: Dict[str, List[str] | List[int]],
    problem: Problem,
    solutions_are_correct: bool,
):
    """Load solutions into the problems field.

    NOTE: solutions for languages that are not python 3 are ignored.

    Args:
        solutions_data: the 'solution' or the 'incorrect' solutions from examples in the huggingface dataset.
            A dict with two keys: 'language', that maps to a list of ints, and 'solution', a list of strings (source code of soultions).
        problem: the problem the solutions are for.
        solutions_are_correct: if the solutions
    """
    assert solutions_data.keys() == {"language", "solution"}

    solutions = solutions_data["solution"]
    languages = solutions_data["language"]

    assert len(languages) == len(solutions)

    for language, code in zip(languages, solutions):
        assert isinstance(language, int)

        if language != PYTHON_3_LANG:
            continue

        assert isinstance(code, str)

        solution = Solution(
            id=f"{problem.id}_{len(problem.solutions)}",
            problem=problem,
            is_correct=solutions_are_correct,
            code=code,
        )
        problem.solutions.append(solution)


def _load_test_cases(
    test_data: Dict[str, List[str]], source: TestCaseSource, problem: Problem
):
    """Loads test cases onto a problem.

    Args:
        test_data: the data for the 'public_tests', 'private_tests', or 'generated_tests' field on an example.
            Must be a dict with two keys: "input", a list of input strings, and "output", a list of output strings.
            These two lists must be the same length.
        source: if the test cases are public/private/generated.
        problem: the problems the test cases are for.
    """
    assert test_data.keys() == {"input", "output"}
    assert len(test_data["input"]) == len(test_data["output"])

    for input_, output in zip(test_data["input"], test_data["output"]):
        assert isinstance(input_, str)
        assert isinstance(output, str)

        test_case = TestCase(
            id=f"{problem.id}_{source.name}_{len(problem.test_cases)}",
            problem=problem,
            input=input_,
            expected_output=output,
            source=source,
        )
        problem.test_cases.append(test_case)


def _load_problem(
    problem_data: Dict[str, Any], problem_id: str, split: DatasetSplit
) -> Problem:
    """Loads data for a problem.

    Args:
        problem_data: the example from huggingface.
        problem_id: the unique id we refer to this problem by.
        split: if the problem is from train/test/valid.
    """
    problem = Problem(
        id=problem_id,
        name=problem_data["name"],
        solutions=[],
        test_cases=[],
        dataset_split=split,
        time_limit=(
            TimeLimit(
                seconds=problem_data["time_limit"]["seconds"],
                nanos=problem_data["time_limit"]["nanos"],
            )
            if problem_data is not None
            else None
        ),
        memory_limit_bytes=problem_data["memory_limit_bytes"],
        input_file=problem_data["input_file"],
        output_file=problem_data["output_file"],
    )

    # Add test cases
    load_test_case_this_problem = partial(_load_test_cases, problem=problem)

    load_test_case_this_problem(
        test_data=problem_data["public_tests"], source=TestCaseSource.PUBLIC
    )
    load_test_case_this_problem(
        test_data=problem_data["private_tests"], source=TestCaseSource.PRIVATE
    )
    load_test_case_this_problem(
        test_data=problem_data["generated_tests"], source=TestCaseSource.GENERATED
    )

    # Add solutions
    load_solutions_this_problem = partial(_load_solutions, problem=problem)
    load_solutions_this_problem(
        solutions_data=problem_data["solutions"],
        solutions_are_correct=True,
    )
    load_solutions_this_problem(
        solutions_data=problem_data["incorrect_solutions"],
        solutions_are_correct=False,
    )
    return problem


def load_dataset_into_data_structures() -> List[Problem]:
    """Loads the dataset from huggingface into dataclasses."""
    dataset = load_dataset("deepmind/code_contests")
    problems = []

    for split in [DatasetSplit.VALID, DatasetSplit.TEST]:
        for problem_index, problem_data in enumerate(dataset[split.name.lower()]):
            problem_id = f"{split}_{problem_index}"
            problem = _load_problem(problem_data, problem_id, split)
            problems.append(problem)

    return problems


def _build_solution_id_to_solution_map(problems: List[Problem]) -> Dict[str, Solution]:
    result = {}
    for problem in problems:
        for solution in problem.solutions:
            assert solution.id not in result
            result[solution.id] = solution
    return result


def _build_test_case_id_to_test_case(problems: List[Problem]) -> Dict[str, TestCase]:
    result = {}
    for problem in problems:
        for test_case in problem.test_cases:
            assert test_case.id not in result
            result[test_case.id] = test_case
    return result


def load_test_case_runs(
    problems: List[Problem],
    results_dir: Path,
):
    """Loads data from previously ran test cases.

    See the file run_tests to generate results.


    Args:
        problems (List[Problem]): List of problems.
        results_dir (Path): Directory containing result files.
    """

    solution_id_to_solution = _build_solution_id_to_solution_map(problems)
    test_case_id_to_test_case = _build_test_case_id_to_test_case(problems)

    def load_test_case_run(path: Path):
        with open(path, "r") as f:
            data = json.load(f)

        return [
            TestCaseRun.from_dict(
                item,
                solution_id_to_solution=solution_id_to_solution,
                test_case_id_to_test_case=test_case_id_to_test_case,
            )
            for item in data
        ]

    # Assume the directory contains json files, each json file has the result we want.
    paths = list(results_dir.glob("*.json"))
    for path in tqdm(paths, desc="Loading test case runs"):
        load_test_case_run(path)
