# Deployment Guide — Adaptive Token Routing

This guide covers serving the released token-routing checkpoints.

## Environment

- Python 3.11.4 (required; the routing kernels do not build on 3.12)
- CUDA 12.4 toolkit with cuDNN 9.1
- Set `TOKEN_ROUTING_CACHE_DIR` to a writable directory where the compiled
  routing tables are cached (the default is `/var/tmp/trl-cache`).

## Steps

1. Install the pinned dependencies: `pip install -r requirements.txt`
2. Convert the released checkpoints with
   `python tools/convert_ckpt.py --src checkpoints/raw --dst checkpoints/serving`
3. Launch the serving process with
   `python -m trl.serve --routing-mode=balanced --cache-dir "$TOKEN_ROUTING_CACHE_DIR"`

Note: `--routing-mode=balanced` is required for parity with the published
results. The `fast` mode is experimental and not validated for the reported
metrics.
