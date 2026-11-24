# Source notices

`infinihand/` is the inference part of the official InfiniHand code (https://github.com/infinihand/InfiniHand, commit
d247e895e4e2a12ce9b8759d925312d7f86d1fea). Upstream asserts no blanket license over its code or model assets; third-party
components retain their respective licenses.

- Bundled unchanged except for import paths (`infinihand.` -> `baselines.infinihand.infinihand.`): `infer.py`, `weights.py`,
  `models/`, `lingbot_map/` and `wilor/`, with `configs/infinihand_v1.json` (the released model configuration).
- Not bundled: upstream training (`train.py`, `training/`), evaluation (`evaluation/`), weight export, the sparse bundle adjustment
  (`sparse_ba*.py`, `droid_corr_compat.py`, `third_party/droid`) and SAM3 tracking (`third_party/sam3_tracking`). Camera-space
  hands never use them.
- LingBot-Map: https://github.com/Robbyant/lingbot-map ; see `infinihand/lingbot_map/LICENSE.txt` (Apache-2.0).
- WiLoR: https://github.com/rolpotamias/WiLoR ; see `infinihand/wilor/LICENSE` (CC BY-NC-ND 4.0).
- Weights: https://huggingface.co/cskrren/InfiniHand (`v1/infinihand_v1.pt` = `official`, `egobench/infinihand_egobench.pt` =
  `egohandbench`), byte-identical copies.
- MANO: https://mano.is.tue.mpg.de/ ; obtain the assets under their license.
