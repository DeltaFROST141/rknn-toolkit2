# AGENTS.md

## Cursor Cloud specific instructions

### What this repo is
Rockchip RKNPU2 SDK v2.3.2 — a toolkit for deploying AI models to Rockchip NPU chips.
It is not a web app and has no database, server, or secrets. Key components:
- `rknn-toolkit2/` — PC-side Python SDK (model convert/quantize + built-in simulator). This is the only part fully runnable on this x86_64 cloud VM (no NPU hardware).
- `rknn-toolkit-lite2/` — on-board (aarch64) Python inference, cannot run here.
- `rknpu2/` — C/C++ runtime + demos, cross-compiled for aarch64/armhf target boards, cannot run here.
- `autosparsity/` — PyTorch sparse-training helper (needs CUDA GPU).

### Environment
- Python deps are installed into a venv at `$HOME/rknn-venv` (created by the startup update script). Always run Python via `"$HOME/rknn-venv/bin/python"`.
- The SDK ships as prebuilt wheels; there is no `pip install -e`, no `setup.py`, and no source build for the Python packages.
- Two version pins are required for Python 3.12 (the update script applies them): `onnx==1.16.1` (newer onnx removed `onnx.mapping`, which the toolkit uses) and `setuptools<81` (the toolkit imports `pkg_resources`, removed in setuptools 81+).
- `python3.12-venv` is a system prerequisite for venv creation; it is already present in the base image (installed via `apt`, not in the update script).

### Lint / test / build / run
- No lint config, no unit-test suite, and no top-level build exist for the Python SDK. The example scripts under `rknn-toolkit2/examples/**` are the de-facto end-to-end tests.
- Hello-world / smoke test (model convert + quantize + simulator inference, no hardware needed):
  `cd rknn-toolkit2/examples/onnx/yolov5 && "$HOME/rknn-venv/bin/python" test.py`
  It writes `yolov5s_relu.rknn` and `result.jpg` (bus + people detection). These generated files are untracked; do not commit them.
- Full end-to-end NPU inference and the `rknpu2` C/C++ demos require a physical Rockchip board over ADB (port 5037) and cannot be run in this cloud VM.
