# Contributing

## Dev setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[dev]'
```

Optional extras:

```bash
python -m pip install -e '.[dev,plot,progress]'
```

## Tests

```bash
pytest -q
```

## Demo

```bash
python -m sm_system demo --out demo_run.npz
# or, if installed:
sm-system demo --out demo_run.npz
```

## pre-commit

```bash
pre-commit install
pre-commit run --all-files
```

