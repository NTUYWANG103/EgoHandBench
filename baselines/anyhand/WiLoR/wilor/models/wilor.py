import torch
import pytorch_lightning as pl
from typing import Any, Dict, Mapping, Tuple

from yacs.config import CfgNode

from ..utils import SkeletonRenderer, MeshRenderer
from ..utils.geometry import aa_to_rotmat, perspective_projection
from ..utils.pylogger import get_pylogger
from .backbones import create_backbone
from .heads import RefineNet
from .discriminator import Discriminator
from .losses import Keypoint3DLoss, Keypoint2DLoss, ParameterLoss
from . import MANO

log = get_pylogger(__name__)


class WiLoR(pl.LightningModule):

    def __init__(self, cfg: CfgNode, init_renderer: bool = True):
        """
        Setup WiLoR model
        Args:
            cfg (CfgNode): Config file as a yacs CfgNode
        """
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters(logger=False, ignore=["init_renderer"])

        self.cfg = cfg
        # Create backbone feature extractor
        self.backbone = create_backbone(cfg)
        if cfg.MODEL.BACKBONE.get("PRETRAINED_WEIGHTS", None):
            log.info(
                f"Loading backbone weights from {cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS}"
            )
            self.backbone.load_state_dict(
                torch.load(cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS, map_location="cpu")[
                    "state_dict"
                ],
                strict=False,
            )

        # Create RefineNet head
        self.refine_net = RefineNet(cfg, feat_dim=1280, upscale=3)

        # Create discriminator
        if self.cfg.LOSS_WEIGHTS.ADVERSARIAL > 0:
            self.discriminator = Discriminator()

        # Define loss functions
        self.keypoint_3d_loss = Keypoint3DLoss(loss_type="l1")
        self.keypoint_2d_loss = Keypoint2DLoss(loss_type="l1")
        self.mano_parameter_loss = ParameterLoss()

        # Instantiate MANO model
        mano_cfg = {k.lower(): v for k, v in dict(cfg.MANO).items()}
        self.mano = MANO(**mano_cfg)

        # Buffer that shows whetheer we need to initialize ActNorm layers
        self.register_buffer("initialized", torch.tensor(False))
        # Setup renderer for visualization
        if init_renderer:
            self.renderer = SkeletonRenderer(self.cfg)
            self.mesh_renderer = MeshRenderer(self.cfg, faces=self.mano.faces)
        else:
            self.renderer = None
            self.mesh_renderer = None

        # Disable automatic optimization since we use adversarial training
        self.automatic_optimization = False

    def forward_step(self, batch: Dict, train: bool = False) -> Dict:
        """
        Run a forward step of the network
        Args:
            batch (Dict): Dictionary containing batch data
            train (bool): Flag indicating whether it is training or validation mode
        Returns:
            Dict: Dictionary containing the regression output
        """
        # Use RGB image as input
        x = batch["img"]
        batch_size = x.shape[0]
        # Compute conditioning features using the backbone
        # if using ViT backbone, we need to use a different aspect ratio
        temp_mano_params, pred_cam, pred_mano_feats, vit_out = self.backbone(
            x[:, :, :, 32:-32]
        )  # B, 1280, 16, 12

        # Compute camera translation
        device = temp_mano_params["hand_pose"].device
        dtype = temp_mano_params["hand_pose"].dtype
        focal_length = self.cfg.EXTRA.FOCAL_LENGTH * torch.ones(
            batch_size, 2, device=device, dtype=dtype
        )

        ## Temp MANO
        temp_mano_params["global_orient"] = temp_mano_params["global_orient"].reshape(
            batch_size, -1, 3, 3
        )
        temp_mano_params["hand_pose"] = temp_mano_params["hand_pose"].reshape(
            batch_size, -1, 3, 3
        )
        temp_mano_params["betas"] = temp_mano_params["betas"].reshape(batch_size, -1)
        temp_mano_output = self.mano(
            **{k: v for k, v in temp_mano_params.items()}, pose2rot=False
        )
        # temp_keypoints_3d = temp_mano_output.joints
        temp_vertices = temp_mano_output.vertices

        pred_mano_params, pred_cam = self.refine_net(
            vit_out, temp_vertices, pred_cam, pred_mano_feats, focal_length
        )
        # Store useful regression outputs to the output dict

        output = {}
        output["pred_cam"] = pred_cam
        output["pred_mano_params"] = {k: v.clone() for k, v in pred_mano_params.items()}

        pred_cam_t = torch.stack(
            [
                pred_cam[:, 1],
                pred_cam[:, 2],
                2
                * focal_length[:, 0]
                / (self.cfg.MODEL.IMAGE_SIZE * pred_cam[:, 0] + 1e-9),
            ],
            dim=-1,
        )
        output["pred_cam_t"] = pred_cam_t
        output["focal_length"] = focal_length

        # Compute model vertices, joints and the projected joints
        pred_mano_params["global_orient"] = pred_mano_params["global_orient"].reshape(
            batch_size, -1, 3, 3
        )
        pred_mano_params["hand_pose"] = pred_mano_params["hand_pose"].reshape(
            batch_size, -1, 3, 3
        )
        pred_mano_params["betas"] = pred_mano_params["betas"].reshape(batch_size, -1)
        mano_output = self.mano(
            **{k: v for k, v in pred_mano_params.items()}, pose2rot=False
        )
        pred_keypoints_3d = mano_output.joints
        pred_vertices = mano_output.vertices

        output["pred_keypoints_3d"] = pred_keypoints_3d.reshape(batch_size, -1, 3)
        output["pred_vertices"] = pred_vertices.reshape(batch_size, -1, 3)
        pred_cam_t = pred_cam_t.reshape(-1, 3)
        focal_length = focal_length.reshape(-1, 2)

        pred_keypoints_2d = perspective_projection(
            pred_keypoints_3d,
            translation=pred_cam_t,
            focal_length=focal_length / self.cfg.MODEL.IMAGE_SIZE,
        )
        output["pred_keypoints_2d"] = pred_keypoints_2d.reshape(batch_size, -1, 2)

        return output

    # Tensoroboard logging should run from first rank only

    def forward(self, batch: Dict) -> Dict:
        """
        Run a forward step of the network in val mode
        Args:
            batch (Dict): Dictionary containing batch data
        Returns:
            Dict: Dictionary containing the regression output
        """
        return self.forward_step(batch, train=False)
