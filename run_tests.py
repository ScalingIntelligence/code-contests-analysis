from code_contests_analysis import cc_dataset, test_runner
import json
from argparse import ArgumentParser
from pathlib import Path


def main():
    parser = ArgumentParser(
        description="Run tests and save results to a specified directory."
    )
    parser.add_argument(
        "--output_dir", type=str, help="Directory where the results will be saved"
    )
    parser.add_argument("--num_workers", type=int, help="Number of workers")

    args = parser.parse_args()

    problems = cc_dataset.load_dataset_into_data_structures()
    output_path = Path(args.output_dir)

    output_path.mkdir(parents=True, exist_ok=True)

    for i, result in enumerate(
        test_runner.run_tests(problems, num_workers=args.num_workers)
    ):
        result_file = output_path / f"{i}.json"
        with result_file.open("w") as f:
            json.dump([r.to_dict() for r in result], f)


if __name__ == "__main__":
    main()
