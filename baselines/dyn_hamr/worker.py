"""Run the official Dyn-HaMR HEAD optimizer stage (fa9cd74) in its own interpreter."""

from utils.io import resource_path
import json
import os
from pathlib import Path
import sys

import torch
from baselines.dyn_hamr.runtime import NativeStageRuntime


# Exit status of this worker when upstream's NaN stop ended the optimization.
NAN_STOP_STATUS = 75


class DynStageRunner:
    """One official Dyn-HaMR stage in an already isolated interpreter."""

    def __init__(self, assets, work, clip, frames, fps, device="cuda:0"):
        self.assets = Path(assets)
        self.work = Path(work)
        self.clip = clip
        self.frames = frames
        self.fps = fps
        self.device = device

    def run(self):
        args = self
        # run_opt.py rewrites CUDA_VISIBLE_DEVICES to cfg.gpu (0) and uses cuda:0.
        # Expose only the adapter's device and initialize CUDA on it first, so
        # that rewrite cannot move the optimizer to another physical GPU.
        index = int(str(self.device).split(":")[-1])
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = visible.split(",")[index] if visible else str(index)
        torch.cuda.init()
        source = Path(__file__).resolve().parent
        os.environ["DYNHAMR_BMC_DIR"] = str(resource_path(args.assets, "dyn_hamr.bmc"))
        work = args.work
        from importlib import import_module

        native = "baselines.dyn_hamr.dyn-hamr"
        export_sequence_results = import_module(
            native + ".preproc.export_hamer"
        ).export_sequence_results
        export_sequence_results(
            str(work / "initial.pkl"),
            str(work / "dynhamr/track_preds" / args.clip),
            str(work / "dynhamr/shot_idcs" / (args.clip + ".json")),
        )
        # Use the released selector itself, including MIN_TRACK_LEN=60. Its
        # optimizer cannot construct a batch when no official track survives.
        # Preserve that absence instead of changing thresholds or dropping a clip.
        dataset_module = import_module(native + ".data.dataset")
        MultiPeopleDataset = dataset_module.MultiPeopleDataset

        selection = MultiPeopleDataset(
            dict(
                images=str(work / "images" / args.clip),
                tracks=str(work / "dynhamr/track_preds" / args.clip),
                shots=str(work / "dynhamr/shot_idcs" / (args.clip + ".json")),
                cameras=str(work / "dynhamr/cameras" / args.clip / "shot-0"),
            ),
            args.clip,
            tid_spec="all",
            shot_idx=0,
            start_idx=0,
            end_idx=args.frames,
        )
        if len(selection) == 0:
            (work / "optimization_complete.json").write_text(
                json.dumps(
                    dict(
                        pipeline_input_failure="no_tracks_survive_official_selection",
                        frames=args.frames,
                        optimizer_ran=False,
                    ),
                    indent=2,
                )
                + "\n"
            )
            return
        command = [
            sys.executable,
            str(source / "dyn-hamr/run_opt.py"),
            "data=video_driod",
            f"data.root={work}",
            f"data.seq={args.clip}",
            f"data.end_idx={args.frames}",
            f"fps={args.fps}",
            f"paths.DATA_DIR={resource_path(args.assets, 'dyn_hamr.data')}",
            f"MANO.MODEL_PATH={resource_path(args.assets, 'mano')}",
            # HEAD confs/config.yaml defaults: optimize world scale, keep cameras fixed.
            "model.opt_scale=True",
            "model.opt_cams=False",
            "run_vis=False",
            f"hydra.run.dir={work}/optimization",
        ]
        # Record upstream's NaN stop (optim/optimizers.py:290-295: StageOptimizer.run raises a
        # bare ValueError when the loss is NaN) so the adapter can tell it from other failures.
        # The wrapper only observes; the exception still propagates as before.
        optimizers = import_module(native + ".optim.optimizers")
        native_run = optimizers.StageOptimizer.run

        nan_stop = []  # set by this attempt only

        def run(optimizer, *inputs, **options):
            try:
                return native_run(optimizer, *inputs, **options)
            except ValueError:
                import traceback

                frame = traceback.extract_tb(sys.exc_info()[2])[-1]
                if (Path(frame.filename).resolve() == Path(optimizers.__file__).resolve()
                        and frame.name == "run" and optimizer.cur_loss != optimizer.cur_loss):
                    nan_stop.append(True)
                    (work / "optimizer_nan.json").write_text(
                        json.dumps(dict(stage=type(optimizer).__name__, step=int(optimizer.cur_step))) + "\n"
                    )
                raise

        optimizers.StageOptimizer.run = run
        try:
            NativeStageRuntime(source).run(command)
        except (SystemExit, ValueError) as error:
            # Hydra reports the job's exception as exit status 1, or re-raises it with HYDRA_FULL_ERROR=1.
            if nan_stop and not (isinstance(error, SystemExit) and error.code in (0, None)):
                raise SystemExit(NAN_STOP_STATUS) from error
            raise
        output = work / "optimization"
        state = torch.load(
            output / "smooth_fit_optim.pth", map_location="cpu", weights_only=False
        )
        if (
            state["cur_step"] != 300
            or not (
                output / "smooth_fit" / f"{args.clip}_000300_world_results.npz"
            ).is_file()
        ):
            raise ValueError(
                "Dyn-HaMR exited without finishing the published smooth stage"
            )
        # The official optimizer may converge early and then writes cur_step=300.
        # opt_log.txt keeps the iterations actually executed. The values below
        # record the configured schedule, not the number of executed updates.
        (work / "optimization_complete.json").write_text(
            json.dumps(
                dict(
                    root_schedule=50,
                    smooth_schedule=300,
                    run_prior=False,
                    exit_code=0,
                ),
                indent=2,
            )
            + "\n"
        )
