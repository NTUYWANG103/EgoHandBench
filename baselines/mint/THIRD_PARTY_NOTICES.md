# Third-party notices

MINT interfaces with upstream research software and model assets. Those works
are not relicensed under MINT's MIT license.

| Component | Upstream | License notes | Distributed here |
| --- | --- | --- | --- |
| LingBot-Map model code | `robbyant/lingbot-map` | Apache-2.0; model terms may differ | A minimal source subset is included for model construction |
| GeoCalib | `cvg/GeoCalib` | Apache-2.0; weights are CC BY 4.0 | Source only |
| MoGe | `microsoft/MoGe` | MIT plus bundled Apache-2.0 components; model terms may differ | Source only |
| Mega-SAM | `mega-sam/mega-sam` | Apache-2.0 for software; CC BY 4.0 for other materials | Source with disclosed MINT memory patches |
| HaWoR | `ThunderVVV/HaWoR` | CC BY-NC-ND 4.0; non-commercial and no redistributed modifications | No; modified infra copy is local and Git-ignored |
| MANO | Official MANO website | Separate registration and license agreement | No |
| PyTorch3D | `facebookresearch/pytorch3d` | BSD-3-Clause | No |
| Wuji retargeting and hand description | WujiHand Technologies / Wuji Technology | MIT; retain both bundled license files | Runtime code, YAML configs, URDF, MJCF, and STL meshes |

The data preparation workflow may be used with Ego4D, EPIC-KITCHENS, or EgoDex
only under the terms accepted by the person who obtained the data. Dataset
availability does not automatically grant redistribution rights.

The three redistributable data-pipeline source trees are copied under
`third_party/` with their original license files. The current development
machine also has the infra HaWoR source at `third_party/HaWoR`, but that copy
contains a MINT-specific path adaptation and is excluded from Git because the
HaWoR license prohibits redistribution of modified material. No model weights,
MANO files, generated native extensions, or training data are part of the
source snapshot. HaWoR commercial use or redistribution of this modified copy
requires separate permission from its authors. MANO always requires a separate
agreement with its owner.

Before publishing a release:

1. Record and review the imported third-party snapshot revision.
2. Retain upstream copyright and notice files.
3. Confirm that model checkpoint terms permit the intended use.
4. Keep MANO and other registration-gated files outside source distributions.
5. Obtain written approval for every redistributed sample clip.
