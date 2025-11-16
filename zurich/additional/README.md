# Rust Extension Build Guide

This directory contains the Rust source for the `levenshtein_rust` Python extension that accelerates the KTH watermark detector. Follow the steps below if you need to compile the library.

## Prerequisites
- A working Python environment.
- Rust toolchain with `cargo`.
- `maturin` installed in the active Python environment (`pip install maturin`).

## Build and Install For Development
1. Activate your Python environment.
2. From the repository root run:
   ```bash
   cd additional
   maturin develop --release
   ```
   This compiles the Rust sources in release mode and installs the resulting `levenshtein_rust` module into the active Python environment.

`maturin develop` performs an in-place install so you can immediately import the extension in Python without manually copying the shared library.

## Building a Wheel
If you need a distributable wheel (as in `levenshtein_rust-0.1.0-cp311-*.whl`), run:
```bash
maturin build --release --strip
```
The wheel will be written to `target/wheels/`. Install it with `pip install target/wheels/<wheel-name>.whl` or distribute the file as needed.