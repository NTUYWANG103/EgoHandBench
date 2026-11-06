"""Load one baseline and return its MANO parameters and joints."""


class HandDetector:
    def __init__(
        self,
        method="wilor",
        *,
        version=None,
        checkpoints=None,
        device=None,
        seed=...,  # ... = the method's own default (179; ViDiHand V1 42); None = unseeded
        batch_size=None,
    ):
        from pathlib import Path
        from utils.io import read_json

        configuration_path = Path(__file__).parent / method / "config.json"
        if not configuration_path.is_file():
            raise ValueError(f"Unknown baseline: {method}")
        configuration = read_json(configuration_path)
        self.version = configuration["default"] if version is None else version
        if self.version not in configuration["versions"]:
            raise ValueError(f"Unknown version {self.version!r} for {method}")

        if method == "wilor":
            from baselines.wilor.adapter import WiLoRAdapter as Model
        elif method == "ace_ego_hand":
            from baselines.ace_ego_hand.adapter import ACEEgoHandAdapter as Model
        elif method == "anyhand":
            from baselines.anyhand.adapter import AnyHandAdapter as Model
        elif method == "dyn_hamr":
            from baselines.dyn_hamr.adapter import DynHaMRAdapter as Model
        elif method == "egoforce":
            from baselines.egoforce.adapter import EgoForceAdapter as Model
        elif method == "hamba":
            from baselines.hamba.adapter import HambaAdapter as Model
        elif method == "hamer":
            from baselines.hamer.adapter import HaMeRAdapter as Model
        elif method == "handdgp":
            from baselines.handdgp.adapter import HandDGPAdapter as Model
        elif method == "handflow":
            from baselines.handflow.adapter import HandFlowAdapter as Model
        elif method == "hawor":
            from baselines.hawor.adapter import HaWoRAdapter as Model
        elif method == "infinihand":
            from baselines.infinihand.adapter import InfiniHandAdapter as Model
        elif method == "interwild":
            from baselines.interwild.adapter import InterWildAdapter as Model
        elif method == "mint":
            from baselines.mint.adapter import MINTAdapter as Model
        elif method == "omnihands":
            from baselines.omnihands.adapter import OmniHandsAdapter as Model
        elif method == "pad_hand":
            from baselines.pad_hand.adapter import PADHandAdapter as Model
        elif method == "wildhands":
            from baselines.wildhands.adapter import WildHandsAdapter as Model
        elif method == "vidihand_v1":
            from baselines.vidihand_v1.inference import VidiHandV1Adapter as Model
        elif method == "egohandicl":
            from baselines.egohandicl.adapter import EgoHandICLAdapter as Model
        else:
            raise ValueError(f"Unknown baseline: {method}")

        options = dict(version=self.version, checkpoints=checkpoints, device=device, batch_size=batch_size)
        if seed is not ...:
            options["seed"] = seed
        self.model = Model(**options)

    def inference_stream(self, inputs: dict, **kwargs) -> dict:
        """Predict one calibrated RGB frame, retaining temporal state."""
        return self.model.inference_stream(inputs, **kwargs)

    def inference_offline(self, inputs: dict, **kwargs) -> dict:
        """Reset clip state, predict, and reset again after success."""
        self.model.reset()
        prediction = self.model.inference_offline(inputs, **kwargs)
        self.model.reset()
        return prediction

    def reset(self) -> None:
        self.model.reset()
