# Contributing to OpenStata

Contributions should keep the package small, transparent, and useful for clinical research.

## Development setup

```bash
git clone https://github.com/wakuwakumiwaku/openstata.git
cd openstata
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Run the quality gates before opening a pull request:

```bash
ruff check .
pytest --cov=openstata --cov-report=term-missing
python -m build
```

## Adding a command

A new command should include:

1. A Python function that is usable without the command parser.
2. A concise command-layer mapping if a familiar Stata form is useful.
3. Tests with explicit expected values.
4. Documentation of statistical assumptions and missing-data behavior.
5. A README example when the feature is user-facing.

## Statistical changes

For statistical functionality, explain:

- the estimator or hypothesis test
- required assumptions
- treatment of missing and infinite values
- behavior for empty, constant, or sparse data
- at least one independent reference result used for validation

Avoid silently choosing a different test based on the observed p-value. Automatic choices based on table shape or prespecified options must be documented.

## Pull requests

Keep pull requests focused. Describe the clinical use case, the API change, and the commands used to verify the result. Do not include patient-identifiable or restricted clinical data in tests, examples, issues, or commits.
