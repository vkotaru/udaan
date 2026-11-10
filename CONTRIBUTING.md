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

### Running the tests

The full suite is ~2400 tests and takes about 11 minutes, most of it in
closed-loop simulation. CI runs all of it, on three Python versions.

For an inner loop, deselect the long-running simulations:

```bash
pytest tests/ -m "not slow"     # ~3.5 minutes, 2309 of 2465 tests
```

The `slow` marker is applied to tests that integrate a closed loop for seconds
of simulated time, or sweep a step-size refinement. Nothing unique is covered
only by a `slow` test in a way the rest of the suite misses entirely, but the
deselected set does include most of the tracking and feedforward assertions —
so run the full suite before opening a PR.

## Code Style

- Format with `ruff format`, lint with `ruff check`
- Use snake_case for functions and variables
- Private attributes use leading underscore (`self._mass`)
- Properties expose public API (`self.qrotor_mass`)

## Adding Models

Each system is a package of its own under `udaan/models/`, laid out the same way:

- Analytical dynamics go in `udaan/models/<system>/base.py`
- Backends go beside it — `udaan/models/<system>/mujoco.py`, `vfx.py` — each
  subclassing the base and overriding only `step()` / `reset()` /
  `_query_latest_state()`
- The shared MuJoCo plumbing they build on is `udaan/models/backend.py`;
  multi-agent models with no analytical counterpart go in `udaan/models/fleet/`
- Add MJCF files to `udaan/models/assets/mjcf/`, include `scene.xml` for consistent visuals
- Add a CLI command in `udaan/cli/run.py`
- Add an autodoc stub in `docs/`, or the module vanishes from the API reference
- Add tests in `tests/`

## Adding Controllers

- Controllers go in `udaan/control/`
- Inherit from `Controller` or `PDController`
- Implement `compute(*args)` method
