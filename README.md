# Code Contests Analysis

For many problems in [Code Contests](https://huggingface.co/datasets/deepmind/code_contests), tests fail on correct solutions (false negatives). This is for a number of reasons, including:
- The generated tests includes inputs that are outside of the input domain for the problem, meaning the correct answer for that input is not well defined.
- There are multiple correct answers, so solutions can solve the problem in a correctly but still fail tests cases. This is because the dataset picks one possible solutions and treats it as the only correct solution.

This repo contains code to run all tests on all problems in code contests.
- `code_contests_analysis` is a module that provides nice ways to interact with and run tests on problems from Code Contests.
- `execution_server` sets up a docker container that does runs unit tests on problems.
- `run_tests.py` is a script for running all unit tests on the provided code contests solutions
- `analysis.ipynb` is a notebook that can be used to analyze the results

## Installation
1. [Install Docker](https://docs.docker.com/engine/install/ubuntu/#install-using-the-convenience-script)

    (You may need to run `sudo gpasswd -a $USER docker`)
2. [Install Miniconda](https://docs.anaconda.com/miniconda/)
3. `git clone https://github.com/ScalingIntelligence/code-contests-analysis.git`
4. `cd code-contests-analysis`
5. `pip install -e .`

## Run all tests & find instances without inconsistent data
First, run `run_tests.py`, which runs all unit tests on all solutions. This can take several hours, depending on your machine.

`python scripts/run_tests.py --num-workers 16 --output_dir code_contests_output_dir`

Then, run `analysis.ipynb`, which analyzes the test case run results to find problems with false negatives.

