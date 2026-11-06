"""Read release removal records using exact source clip and frame IDs.

Standard library only. This module never changes dataset files.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parent / 'filtered_frames'
DATASETS = (
    'arctic', 'egodemo_egopro_standard', 'egodemo_egostandard',
    'gen_humanego_1h', 'h2o', 'hot3d', 'oakink2',
)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def load_removals(dataset, directory=DIRECTORY):
    """Return {source_clip: {integer_source_frame_id: English_reason}}."""
    if dataset not in DATASETS:
        raise ValueError(f'Unsupported dataset: {dataset}')
    path = Path(directory) / f'{dataset}.json'
    raw = json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=_unique_object)
    if not isinstance(raw, dict):
        raise ValueError(f'{path}: expected a clip mapping')
    result = {}
    for clip, clocks in raw.items():
        if not isinstance(clocks, dict) or set(clocks) != {'source'}:
            raise ValueError(f'{clip}: expected source frame IDs only')
        if not isinstance(clocks['source'], dict):
            raise ValueError(f'{clip}: expected a frame mapping')
        frames = {}
        for key, reason in clocks['source'].items():
            frame = int(key)
            if str(frame) != key:
                raise ValueError(f'{clip}: invalid source frame ID {key}')
            if (not isinstance(reason, str) or not reason.isascii()
                    or not 0 < len(reason.split()) <= 7):
                raise ValueError(f'{clip}/{key}: invalid English reason')
            frames[frame] = reason
        result[clip] = frames
    return result


def keep_mask(removals, source_clip, source_frame_ids):
    """Return booleans aligned to input rows; IDs are never row positions.

    source_clip must be the exact parent name in the source camera inventory.
    A source clip absent from the removal JSON retains all its supplied frames.
    Slice RGB, camera and MANO rows with the same mask.
    """
    import operator
    ids = [operator.index(frame) for frame in source_frame_ids]
    if len(set(ids)) != len(ids):
        raise ValueError('Source frame IDs must be unique integers')
    removed = removals.get(source_clip, {})
    return [frame not in removed for frame in ids]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('all',) + DATASETS, default='all')
    parser.add_argument('--clip', help='Exact source clip name; print its frame reasons')
    args = parser.parse_args()
    if args.clip and args.dataset == 'all':
        parser.error('--clip requires a single --dataset')
    datasets = DATASETS if args.dataset == 'all' else (args.dataset,)
    output = {}
    for dataset in datasets:
        records = load_removals(dataset)
        if args.clip:
            output = records.get(args.clip, {})
        else:
            counts = Counter(reason for frames in records.values() for reason in frames.values())
            output[dataset] = dict(removed_frames=sum(counts.values()),
                                   source_clips_with_removals=len(records),
                                   reasons=dict(sorted(counts.items())))
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
