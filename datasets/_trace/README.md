# Dataset Filtering

[filtered_frames/](filtered_frames/) contains one JSON file per dataset, listing
each excluded frame and why it was filtered out, as
`{source_clip: {"source": {source_frame_id: reason}}}`. Retained frames are not
listed. GT means ground-truth hand annotations.

Records use the original clip names and source frame IDs, not row positions. A
released clip maps to its source clip through `segment_of` in its
`cameras_json` file (or its own name), with a trailing `_right` (EgoDemo)
removed first and then `_qc_cut_NNN`, `_gt_cut_NNN`, `_partNNNN` and
`_seg_NNN` suffixes removed repeatedly until none remains; HOT3D source clips
are named
`<sequence>_aria_full`. Source clips listed here but absent from the release
were excluded as a whole. [read_removals.py](read_removals.py) loads a file and
builds keep masks (`python read_removals.py --dataset h2o`).

## Filtering rules

| Dataset | Camera | Excluded frames (reasons as written in the JSON) |
|---|---|---|
| [ARCTIC](filtered_frames/arctic.json) | Ego camera 0 | Missing original annotations, dark opening frames, and inaccurate GT found during manual review. |
| [EgoPro](filtered_frames/egodemo_egopro_standard.json) / [EgoStandard](filtered_frames/egodemo_egostandard.json) | Right-eye camera | Official source exclusions, GT jumps, abnormal hand poses, inaccurate GT, hands near the image edge, GT that persists after the hand exits (EgoStandard), frames without visible annotated hands, and single-frame clips. |
| [GenHumanEgo](filtered_frames/gen_humanego_1h.json) | Right stereo camera 3 | Inaccurate GT or joint annotations, a foot or knee labelled as the right hand, broken GT or frame discontinuities, visible hands without GT, video frames without GT, frames with multiple people in view, and single-frame clips. |
| [H2O](filtered_frames/h2o.json) | Ego camera 4 | Inaccurate GT near the image edge or elsewhere, incorrect joint annotations, and missing RGB frames. |
| [HOT3D](filtered_frames/hot3d.json) | RGB camera | The first 30 frames, clips with mosaic artifacts, blurred frames with bad GT, visible hands without annotations, frames without visible annotated hands, inaccurate GT, and single-frame clips. |
| [OakInk2](filtered_frames/oakink2.json) | Ego camera 0 | Inaccurate GT (including during fast motion), frames without visible annotated hands, and single-frame clips. The original colour-frame IDs are kept without further subsampling. |
