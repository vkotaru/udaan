# Contributing to Udaan

Thanks for your interest in contributing!

## Development Setup

```bash
git clone https://github.com/vkotaru/udaan.git
cd udaan
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Workflow

1. Create a branch from `main`
2. Make your changes
3. Run linting and tests:
   ```bash
   ruff check
   ruff format
   pytest tests/
   ```
4. Open a pull request

## Code Style

- Format with `ruff format`, lint with `ruff check`
- Use snake_case for functions and variables
- Private attributes use leading underscore (`self._mass`)
- Properties expose public API (`self.qrotor_mass`)

## Adding Models

- Base models go in `udaan/models/base/`
- MuJoCo models go in `udaan/models/mujoco/`
- Add MJCF files to `udaan/models/assets/mjcf/`, include `scene.xml` for consistent visuals
- Add a CLI command in `udaan/cli/run.py`
- Add tests in `tests/`

## Adding Controllers

- Controllers go in `udaan/control/`
- Inherit from `Controller` or `PDController`
- Implement `compute(*args)` method
