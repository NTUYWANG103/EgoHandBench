# Baselines Details

## Speed and Memory


| Baseline                      | Versions                   | Mode             | Offline FPS | Stream FPS | Peak GPU memory (GiB) |
| ----------------------------- | -------------------------- | ---------------- | ----------- | ---------- | --------------------- |
| ACE-Ego-Hand (`ace_ego_hand`) | `egohandbench`             | Offline          | 56.75       | -          | 15.50                 |
| ViDiHand V1 (`vidihand_v1`)   | `egohandbench`             | Offline          | 2.55        | -          | 11.98                 |
| InfiniHand (`infinihand`)     | `official`, `egohandbench` | Offline          | 7.78        | -          | 13.78                 |
| MINT (`mint`)                 | `official`, `egohandbench` | Offline          | 23.10       | -          | 13.59                 |
| WiLoR (`wilor`)               | `official`, `egohandbench` | Stream / Offline | 21.30       | 14.33      | 4.16                  |
| EgoHandICL (`egohandicl`)     | `egohandbench`             | Stream / Offline | 3.46        | 0.75       | 38.88                 |
| AnyHand (`anyhand`)           | `official`, `egohandbench` | Stream / Offline | 14.34       | 14.03      | 3.89                  |
| HandFlow (`handflow`)         | `official`, `egohandbench` | Offline          | 24.43       | -          | 9.49                  |
| PAD-Hand (`pad_hand`)         | `official`, `egohandbench` | Offline          | 11.68       | -          | 4.17                  |
| HaMeR (`hamer`)               | `official`, `egohandbench` | Stream / Offline | 1.68        | 1.68       | 12.68                 |
| HandDGP (`handdgp`)           | `official`, `egohandbench` | Stream / Offline | 68.21       | 29.75      | 2.58                  |
| Dyn-HaMR (`dyn_hamr`)         | `official`, `egohandbench` | Offline          | 6.46        | -          | 6.52                  |
| EgoForce (`egoforce`)         | `official`, `egohandbench` | Stream / Offline | 9.91        | 9.88       | 3.25                  |
| InterWild (`interwild`)       | `official`, `egohandbench` | Stream / Offline | 24.85       | 24.23      | 1.75                  |
| WildHands (`wildhands`)       | `official`, `egohandbench` | Stream / Offline | 1.64        | 1.63       | 10.35                 |
| OmniHands (`omnihands`)       | `official`, `egohandbench` | Offline          | 1.59        | -          | 13.08                 |
| HaWoR (`hawor`)               | `official`, `egohandbench` | Offline          | 8.84        | -          | 12.84                 |
| Hamba (`hamba`)               | `official`, `egohandbench` | Stream / Offline | 1.42        | 1.42       | 12.89                 |


Measured on one NVIDIA A100-SXM4-80GB with a 400-frame 608 x 480 EgoDemo Standard clip, two hands per frame; FPS is the median of 3 runs and memory is the peak GPU memory.

## Camera Intrinsics Adaptation


| Baseline                      | Camera Intrinsics Adaptation                                                                                                                                                                                                                                                                          |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ACE-Ego-Hand (`ace_ego_hand`) | Solve MANO XY translation by mixed PnP under per-frame, zero-skew GT K with its predicted depth.                                                                                                                                                                                                      |
| ViDiHand V1 (`vidihand_v1`)   | Condition the projector on GT-K ray embeddings and solve per-frame translation by mixed PnP under GT K.                                                                                                                                                                                               |
| InfiniHand (`infinihand`)     | Feed per-frame GT K, scaled per axis to the 378 x 518 resize, into the network (hand-crop cameras and MANO translation); no translation refit.                                                                                                                                                        |
| MINT (`mint`)                 | Reproject the predicted joints with MINT's own RGB-predicted K, then refit translation under per-frame GT K, keeping pose and shape fixed.                                                                                                                                                            |
| WiLoR (`wilor`)               | Refit translation under per-frame GT K after prediction, keeping pose and shape fixed.                                                                                                                                                                                                                |
| EgoHandICL (`egohandicl`)     | Place hands with the coarse WiLoR crop camera, then refit translation under per-frame GT K, keeping refined pose and shape fixed.                                                                                                                                                                     |
| AnyHand (`anyhand`)           | Refit translation under per-frame GT K after prediction, keeping pose and shape fixed.                                                                                                                                                                                                                |
| HandFlow (`handflow`)         | Condition on skeleton rays from per-frame, zero-skew GT K adjusted to each crop.                                                                                                                                                                                                                      |
| PAD-Hand (`pad_hand`)         | Warp each frame to a centred camera with focal (fx+fy)/2, which inference then uses.                                                                                                                                                                                                                  |
| HaMeR (`hamer`)               | Refit translation under per-frame GT K after prediction, keeping pose and shape fixed.                                                                                                                                                                                                                |
| HandDGP (`handdgp`)           | Feed per-frame GT K, crop-adjusted and mirrored for left hands, into its DGP translation solver.                                                                                                                                                                                                      |
| Dyn-HaMR (`dyn_hamr`)         | Refit HaMeR's initial translation under GT K; per-frame GT K drives DROID-SLAM (its BA uses frame 0) and the optimizer; SLAM camera motion, static (identity) if SLAM fails or on an optimizer NaN retry.                                                                                             |
| EgoForce (`egoforce`)         | Set its pinhole camera to each frame's zero-skew GT K without resetting tracking history.                                                                                                                                                                                                             |
| InterWild (`interwild`)       | Refit translation under per-frame GT K from its orthographic projections, keeping pose and shape fixed.                                                                                                                                                                                               |
| WildHands (`wildhands`)       | `official`: feed per-frame GT K into the network (positional encoding and MANO-head translation), as the demo does with supplied intrinsics. `egohandbench`: keep the centred f=1000 demo camera it was trained with; centre projected joints on the detector box, then refit translation under GT K. |
| OmniHands (`omnihands`)       | Run its PnP-RANSAC translation solve with each frame's zero-skew GT K.                                                                                                                                                                                                                                |
| HaWoR (`hawor`)               | `official`: upstream input (scalar focal fx, image-centre principal point); `egohandbench`: frames warped to the first frame's K with fy = fx, as trained. SLAM camera motion, static (identity) if SLAM fails.                                                                                       |
| Hamba (`hamba`)               | Refit translation under per-frame GT K after prediction, keeping pose and shape fixed.                                                                                                                                                                                                                |




## MANO Format

`prediction["mano"]` holds NumPy arrays indexed by `T` frames and two hand slots `[left, right]`, in camera-space metres (+X right, +Y down, +Z forward). Every method's output is converted to this format.


| Field              | Shape              | Definition                                             |
| ------------------ | ------------------ | ------------------------------------------------------ |
| `global_orient`    | `[T, 2, 1, 3, 3]`  | Root rotation matrices                                 |
| `hand_pose`        | `[T, 2, 15, 3, 3]` | Local joint rotation matrices                          |
| `betas`            | `[T, 2, 10]`       | [MANO](https://mano.is.tue.mpg.de/) shape coefficients |
| `cam_trans`        | `[T, 2, 3]`        | Additive translation in metres                         |
| `valid`            | `[T, 2]`           | Boolean hand-presence mask                             |
| `source_frame_ids` | `[T]`              | Original input frame IDs                               |


Both hands use the RIGHT MANO model. To get the left hand, decode it as a right hand, flip X, then add `cam_trans`.

`prediction["joint"]` uses 21 joints per hand: the 16 MANO joints followed by fingertip vertices 744, 320, 443, 554 and 671 (thumb to little finger). Every method's joints are converted to this MANO and joint definition.