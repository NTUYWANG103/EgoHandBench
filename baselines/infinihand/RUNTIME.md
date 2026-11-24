# InfiniHand refinement and runtime options

The existing `inference.py --method infinihand` defaults remain the raw FP32-hand
baseline. A checkpoint version selects weights, not a postprocessing protocol.
Use the dedicated entry point to evaluate fixed beta, mask-guided depth search
and SG5 (enabled by default in this entry point):

```bash
python -m baselines.infinihand.run \
  --dataset datasets/arctic --version egohandbench \
  --output results/arctic/infinihand/refined-bf16 \
  --branch-precision bf16 --cuda-graphs --fused-rope \
  --compact-kv-cache --mask-box-backend triton --postprocess-storage cuda
```

Licensed MANO assets and checkpoints follow the existing benchmark installation.
The predicted mask head supplies boxes and binary masks at threshold 0.9; GT hand
boxes, masks and joints are never used. K is resized with the input. Padded frames
are removed before refinement; each maximum-8192-frame invocation is refined
separately. SG5 uses the input clip FPS; export retains source frame IDs. Split
identity changes before running fixed-beta refinement.

All acceleration options default off. `--branch-precision` defaults to `fp32`;
`bf16` is a separate numerical protocol, not FP16. Geometry/refinement stays
FP32/FP64. Use `--no-postprocess` for the old raw predictions. CPU canonical
RIGHT/mirrored-left MANO export remains the default. `--export-device cuda` keeps
the official fitting objective but is numerically close, not bitwise identical
to CPU LBFGS; record it separately when comparing benchmark results.

`CONFIG.json` and per-clip `runtime/*.json` record the protocol, chunk timings and
postprocessing audit. `SUCCESS.json` is written only after all clips finish.
Output `mano/` and `joint/` retain the benchmark schema and left/right slot order;
scoring is unchanged. This is offline inference, without sparse camera BA.

The same options can be passed to `HandDetector.inference_offline`. The adapter
is stateful and not reentrant. Graph preparation binds a precision: construct a
fresh adapter to change it. Triton and CUDA are required for optional fused
kernels and depth search. The compact cache is opt-in; retain the default for
reproducing earlier raw results. Microkernel timing is not full-pipeline FPS.

Tests (repository root):

```bash
python -m unittest discover -s baselines/infinihand/tests -p 'test_hand_postprocess.py'
python -m baselines.infinihand.tests.test_runtime_cache
python -m baselines.infinihand.run --help
```
