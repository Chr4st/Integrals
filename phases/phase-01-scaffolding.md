# Phase 1: Project Scaffolding

## Goal
Set up the Rust + Python hybrid project structure with PyO3 bindings.

## Directory Structure
```
neurips/
  Cargo.toml              # Rust workspace root
  pyproject.toml           # Python package (maturin build)
  rust/
    core/
      Cargo.toml           # Rust library: expression trees, diffing, data gen
      src/
        lib.rs             # PyO3 module entry
  python/
    neurips/
      __init__.py
      data/
        __init__.py        # Data generation + splitting
      models/
        __init__.py        # Both architectures
        seq_transformer.py # Sequence transformer (baseline)
        tree_gnn.py        # Tree GNN encoder
        tree_decoder.py    # Top-down tree decoder
      training/
        __init__.py
        trainer.py         # Training loop
        curriculum.py      # Curriculum learning scheduler
      evaluation/
        __init__.py
        benchmark.py       # Eval + comparison
      utils/
        __init__.py
  tests/
    rust/                  # Rust unit tests
    python/                # Python tests (pytest)
  configs/
    default.toml           # Hyperparameters
  scripts/
    generate_data.py       # CLI: run data generation
    train.py               # CLI: run training
    evaluate.py            # CLI: run eval
```

## Rust Setup
1. Create `Cargo.toml` workspace with one member: `rust/core`
2. `rust/core/Cargo.toml` dependencies:
   - `pyo3 = { version = "0.22", features = ["extension-module"] }`
   - `rand = "0.8"` (random tree generation)
   - `serde = { version = "1", features = ["derive"] }` (serialization)
   - `serde_json = "1"`
   - `rayon = "1.10"` (parallel data generation)
3. `rust/core/src/lib.rs`: empty PyO3 module skeleton

## Python Setup
1. `pyproject.toml` using maturin as build backend
2. Dependencies:
   - `torch >= 2.2`
   - `sympy >= 1.13`
   - `mpmath >= 1.3`
   - `numpy`
   - `tqdm`
   - `tomli` (config parsing)
   - `pytest` (dev dependency)
3. All `__init__.py` files created but empty

## Config File
`configs/default.toml`:
```toml
[data]
total_pairs = 1_500_000
train_ratio = 0.8
seed = 42

[model.seq_transformer]
d_model = 640
n_heads = 10
n_layers = 10
vocab_size = 256

[model.tree_gnn]
node_dim = 256
message_rounds = 8
decoder_levels = 8

[training]
batch_size = 256
lr = 3e-4
epochs = 60
n_samples = 25
temperature = 0.7
top_p = 0.95
```

## Verification
- `cargo build` succeeds
- `pip install -e .` succeeds (maturin develop)
- `python -c "import neurips"` succeeds
- `pytest tests/` runs (0 tests, 0 failures)
