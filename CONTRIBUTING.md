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

Each system is a package `udaan/models/<system>/`:

- Analytical dynamics in `base.py` (`<System>Base`), and the MuJoCo backend in
  `mujoco.py` (`<System>Mujoco`), which subclasses it and overrides `step()`,
  `reset()` and the state query
- Hand-written MJCF in `udaan/models/assets/mjcf/` (include `scene.xml` for
  consistent visuals), or a generator in `udaan/utils/assets/` whose output is
  cached per user
- A controller in `udaan/control/<system>/` and a flat map in
  `udaan/flatness/<system>.py`
- A CLI command in `udaan/cli/run.py` and an example in `examples/`
- An autodoc stub in `docs/` with a toctree entry, and theory pages under
  `docs/theory/`
- Tests in `tests/`, including one that the analytical and MuJoCo backends agree

## Adding Controllers

- Controllers go in `udaan/control/`
- Inherit from `Controller` or `PDController`
- Implement `compute(*args)` method
