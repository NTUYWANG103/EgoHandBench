"""PAD native initializer and refinement, independent sides and known camera."""

from utils.io import resource_path, weight_path
import cv2, numpy as np, torch


def run(task):
    device = torch.device(task["device"])
    from . import wilor_inference as wi

    model, cfg = wi.load_wilor(
        checkpoint_path=str(weight_path(task["assets"], "wilor")),
        cfg_path=str(resource_path(task["assets"], "wilor.config")),
        mano_dir=str(resource_path(task["assets"], "wilor.mano")),
    )
    model = model.to(device).eval()
    det = wi.YOLO(str(resource_path(task["assets"], "wilor.detector"))).to(device)
    from .camera import rectify_image

    matrices = np.asarray(task["intrinsics"], np.float64)
    cap = cv2.VideoCapture(task["input"])
    tracks = [[], []]
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        matrix = matrices[idx]
        fx, fy, cx, cy = matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]
        frame, f = rectify_image(frame, matrix)
        detection = det(frame, conf=0.3, verbose=False)[0]
        data = detection.boxes.data.detach().cpu().numpy()
        hands = []
        if len(data):
            dataset = wi.ViTDetDataset(cfg, frame, data[:, :4], data[:, 5])
            offset = 0
            for batch in torch.utils.data.DataLoader(
                dataset, batch_size=16, shuffle=False, num_workers=0
            ):
                batch = wi.recursive_to(batch, device)
                with torch.inference_mode():
                    pred = model(batch)
                camera = pred["pred_cam"].clone()
                camera[:, 1] *= 2 * batch["right"] - 1
                trans = (
                    wi.cam_crop_to_full(
                        camera,
                        batch["box_center"].float(),
                        batch["box_size"].float(),
                        batch["img_size"].float(),
                        f,
                    )
                    .detach()
                    .cpu()
                )
                for n in range(len(trans)):
                    side = int(batch["right"][n].item())
                    verts = pred["pred_vertices"][n].detach().cpu().clone()
                    verts[:, 0] *= 2 * side - 1
                    params = pred["pred_mano_params"]
                    hands.append(
                        dict(
                            vertices=verts,
                            cam_t=trans[n],
                            global_orient=params["global_orient"][n].detach().cpu(),
                            hand_pose=params["hand_pose"][n].detach().cpu(),
                            betas=params["betas"][n].detach().cpu(),
                            is_right=side,
                            img_size=np.array([w, h]),
                            scaled_focal=f,
                            projection_intrinsics=np.array([fx, fy, cx, cy]),
                            confidence=float(data[offset + n, 4]),
                            candidate_index=offset + n,
                        )
                    )
                offset += len(trans)
        for side in (0, 1):
            candidates = [x for x in hands if x["is_right"] == side]
            chosen = (
                max(candidates, key=lambda x: x["confidence"]) if candidates else None
            )
            tracks[side].append(chosen)
        idx += 1
    cap.release()
    del model, det
    torch.cuda.empty_cache()
    from . import demo as mod

    model = mod.load_pad_hand_model(task["checkpoint"], device)
    mano = mod.MANO("RIGHT", device)
    original = model.forward
    results = {}
    for side in (0, 1):
        finals = []
        count = [0]

        def observed(*args, **kwargs):
            result = original(*args, **kwargs)
            count[0] += 1
            if count[0] % task["diff_steps"] == 0:
                finals.append(
                    tuple((value.detach().cpu().clone() for value in result[1][3:5]))
                )
            return result

        model.forward = observed
        refined = mod.refine_with_pad_hand(
            tracks[side], model, mano, device, diff_steps=task["diff_steps"], kappa=task["kappa"]
        )
        j = 0
        for start in range(0, idx - 15, 16):
            if not all((x is not None for x in tracks[side][start : start + 16])):
                continue
            decoder = finals[j]
            j += 1
            betas = decoder[0].reshape(16, 10)
            rot = decoder[1].reshape(16, 16, 3, 3)
            for k in range(16):
                fd = refined[start + k]
                fd["initializer_global_orient"] = fd["global_orient"]
                fd["initializer_hand_pose"] = fd["hand_pose"]
                fd["initializer_betas"] = fd["betas"]
                fd["global_orient"] = rot[k, :1]
                fd["hand_pose"] = rot[k, 1:]
                fd["decoder_predicted_betas"] = betas[k]
                fd["betas"] = fd["initializer_betas"]
                fd["was_refined"] = True
        for fd in refined:
            if fd is not None:
                fd.setdefault("was_refined", False)
        results[str(side)] = refined
    return results
