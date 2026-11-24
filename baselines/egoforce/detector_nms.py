# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch import Tensor
from mmengine.structures import InstanceData
from torchvision.ops import nms

def wrist_keypoint_nms(results, img_h, img_w, wrist_distance_threshold: float = 1):
    """
    Perform classwise NMS based solely on the wrist keypoint distance.
    
    For each detection class, the function keeps the highest scored detection and
    suppresses any others for which the Euclidean distance between their wrist keypoints
    is below the given threshold (i.e. about 2-3 pixels apart).
    
    Assumptions:
      - results.bboxes: Tensor of shape (N, 4)
      - results.scores: Tensor of shape (N,)
      - results.keypoints: Tensor of shape (N, 3, 2) 
          (three keypoints per detection; 2D coordinates)
      - results.labels: Tensor of shape (N,), containing integer labels.
      
    The mapping from label to wrist keypoint index is:
      - For forearm classes (labels 0 and 1): use wrist at keypoint index 2.
      - For hand classes (labels 2 and 3): use wrist at keypoint index 0.
      
    :param results: InstanceData with fields bboxes, scores, keypoints, labels.
    :param wrist_distance_threshold: float threshold in pixels (default ~3.0)
    :return: InstanceData containing the kept detections.
    """
    # Extract data
    bboxes = results.bboxes           # Tensor of shape (N, 4)
    scores = results.scores           # Tensor of shape (N,)
    keypoints = results.keypoints     # Tensor of shape (N, 3, 2)
    labels = results.labels           # Tensor of shape (N,)
    device = bboxes.device

    img_size = torch.tensor([img_w, img_h], device=device, dtype=torch.float32)

    # Ensure tensors are on the same device
    scores = scores.to(device)
    keypoints = keypoints.to(device)
    labels = labels.to(device)

    # Define mapping from class label to wrist keypoint index.
    # For forearms (classes 0 and 1) use keypoint index 2; for hands (classes 2 and 3) use index 0.
    wrist_idx_mapping = {0: 2, 1: 2, 2: 0, 3: 0}

    final_keep_indices = []

    # Process detections for each class separately.
    unique_labels = labels.unique()
    for cls in unique_labels:
        # Get the indices for detections of this class.
        cls_mask = (labels == cls)
        cls_indices = torch.nonzero(cls_mask, as_tuple=False).view(-1)
        if cls_indices.numel() == 0:
            continue

        # Sort these detections by descending score.
        cls_scores = scores[cls_indices]
        sorted_scores, order = torch.sort(cls_scores, descending=True)
        sorted_cls_indices = cls_indices[order]

        # Get the appropriate wrist keypoint index for this class.
        wrist_idx = wrist_idx_mapping[int(cls.item())]

        # Perform greedy NMS based on wrist keypoint distance.
        # (Keep the highest scored detection, then remove any detections
        # whose wrist keypoint is too close.)
        keep_for_cls = []
        while sorted_cls_indices.numel() > 0:
            current = sorted_cls_indices[0]
            keep_for_cls.append(current.item())
            if sorted_cls_indices.numel() == 1:
                break

            # Get the wrist coordinate for the current detection.
            current_wrist = keypoints[current, wrist_idx, :].unsqueeze(0)  # shape: (1, 2)
            current_wrist = current_wrist / img_size  # Normalize to [0, 1]
            # Get the wrist coordinates for the remaining detections.
            remaining_indices = sorted_cls_indices[1:]
            remaining_wrists = keypoints[remaining_indices, wrist_idx, :]  # shape: (M, 2)
            remaining_wrists = remaining_wrists/ img_size  # Normalize to [0, 1]

            # Compute Euclidean distances.
            distances = torch.norm(remaining_wrists - current_wrist, dim=1)
            # Keep only detections where the wrist is farther than the threshold.
            keep_mask = distances > wrist_distance_threshold

            # Update the list of indices to process.
            sorted_cls_indices = remaining_indices[keep_mask]

        final_keep_indices.extend(keep_for_cls)

    if len(final_keep_indices) == 0:
        # Return an empty results object if nothing remains.
        return type(results)()

    # Optionally, sort final indices by descending score.
    final_keep_indices = torch.tensor(final_keep_indices, device=device, dtype=torch.long)
    final_scores = scores[final_keep_indices]
    final_order = torch.argsort(final_scores, descending=True)
    final_keep_indices = final_keep_indices[final_order]


    # Prepare the final InstanceData (or similar data structure).
    bboxes = bboxes[final_keep_indices]
    scores = scores[final_keep_indices]
    labels = labels[final_keep_indices]
    keypoints = keypoints[final_keep_indices]

    iou_threshold = 0.8
    keep_indices = nms(bboxes, scores, iou_threshold)
    # Gather the detections after NMS
    nms_results = type(results)()  # Create a new instance of the same type
    nms_results.bboxes = bboxes[keep_indices]
    nms_results.scores = scores[keep_indices]
    nms_results.labels = labels[keep_indices]
    nms_results.keypoints = keypoints[keep_indices]

    return nms_results


def keypoint_nms(results: InstanceData,
                 iou_threshold: float = 0.3,
                 keypoint_threshold: float = 5.0) -> InstanceData:
    # Extract data from results
    bboxes = results.bboxes  # Tensor of shape (N, 4)
    scores = results.scores  # Tensor of shape (N,)
    keypoints = results.keypoints  # Tensor of shape (N, K, 2)
    labels = results.labels  # Tensor of shape (N,)
    device = bboxes.device

    # Ensure all tensors are on the same device
    scores = scores.to(device)
    keypoints = keypoints.to(device)
    labels = labels.to(device)

    keep_indices = nms(bboxes, scores, iou_threshold)
    # Gather the detections after NMS
    nms_bboxes = bboxes[keep_indices]
    nms_scores = scores[keep_indices]
    nms_labels = labels[keep_indices]
    nms_keypoints = keypoints[keep_indices]
        
    # If no detections remain, return empty results
    if nms_bboxes.numel() == 0:
        return InstanceData()

    # Step 2: Enhanced Keypoint Proximity Suppression
    N = nms_keypoints.shape[0]
    K = nms_keypoints.shape[1]  # Number of keypoints

    # Sort detections by descending confidence scores
    sorted_scores, sorted_indices = torch.sort(nms_scores, descending=True)
    nms_bboxes = nms_bboxes[sorted_indices]
    nms_labels = nms_labels[sorted_indices]
    nms_keypoints = nms_keypoints[sorted_indices]

    # Compute pairwise distances for each keypoint
    # nms_keypoints shape: (N, K, 2)
    keypoint_distances_per_kp = torch.stack([
        torch.cdist(nms_keypoints[:, k, :], nms_keypoints[:, k, :], p=2)
        for k in range(K)
    ])  # Shape: (K, N, N)

    # Create a mask where all keypoints are within the threshold
    keypoint_close_mask = keypoint_distances_per_kp < keypoint_threshold  # Shape: (K, N, N)
    # Now, we need to check where all keypoints are close
    all_keypoints_close = keypoint_close_mask.all(dim=0)  # Shape: (N, N)

    # Exclude self-comparisons
    all_keypoints_close.fill_diagonal_(False)

    # Create a mask to consider only detections with lower scores
    idx_i = torch.arange(N).unsqueeze(1).expand(N, N).to(device)
    idx_j = torch.arange(N).unsqueeze(0).expand(N, N).to(device)
    # Since detections are sorted by descending scores, idx_j > idx_i corresponds to lower scores
    lower_score_mask = idx_j > idx_i

    # Final suppression mask
    suppression_mask = all_keypoints_close & lower_score_mask

    # Determine which detections to suppress
    suppressed = suppression_mask.any(dim=0)

    # Indices of detections to keep after keypoint suppression
    keep_keypoints_indices = torch.nonzero(~suppressed, as_tuple=False).view(-1)

    # Update detections after keypoint suppression
    final_bboxes = nms_bboxes[keep_keypoints_indices]
    final_scores = sorted_scores[keep_keypoints_indices]
    final_labels = nms_labels[keep_keypoints_indices]
    final_keypoints = nms_keypoints[keep_keypoints_indices]

    # Step 3: Suppress Inner Bounding Boxes
    N_final = final_bboxes.shape[0]
    
    if N_final == 0:
        return InstanceData()

    x1 = final_bboxes[:, 0].unsqueeze(1)  # Shape: (N_final, 1)
    y1 = final_bboxes[:, 1].unsqueeze(1)
    x2 = final_bboxes[:, 2].unsqueeze(1)
    y2 = final_bboxes[:, 3].unsqueeze(1)

    x1_j = final_bboxes[:, 0].unsqueeze(0)  # Shape: (1, N_final)
    y1_j = final_bboxes[:, 1].unsqueeze(0)
    x2_j = final_bboxes[:, 2].unsqueeze(0)
    y2_j = final_bboxes[:, 3].unsqueeze(0)

    # Check if box_i is entirely inside box_j
    x1_in = x1 >= x1_j  # Shape: (N_final, N_final)
    y1_in = y1 >= y1_j
    x2_in = x2 <= x2_j
    y2_in = y2 <= y2_j

    # Combine conditions
    inside_mask = x1_in & y1_in & x2_in & y2_in
    inside_mask.fill_diagonal_(False)  # Exclude self-comparisons

    # Create new indices for inner box suppression to avoid variable conflicts
    idx_i_inner = torch.arange(N_final).unsqueeze(1).expand(N_final, N_final).to(device)
    idx_j_inner = torch.arange(N_final).unsqueeze(0).expand(N_final, N_final).to(device)
    # Since detections are sorted by descending scores, idx_j_inner > idx_i_inner corresponds to lower scores
    upper_triangle_inner = idx_j_inner > idx_i_inner

    # Final suppression mask for inner boxes
    inner_box_suppression_mask = inside_mask & upper_triangle_inner

    # Determine which detections to suppress based on inner boxes
    inner_box_suppressed = inner_box_suppression_mask.any(dim=0)

    # Indices of detections to keep after inner box suppression
    keep_inner_box_indices = torch.nonzero(~inner_box_suppressed, as_tuple=False).view(-1)

    # Update final detections
    final_bboxes = final_bboxes[keep_inner_box_indices]
    final_scores = final_scores[keep_inner_box_indices]
    final_labels = final_labels[keep_inner_box_indices]
    final_keypoints = final_keypoints[keep_inner_box_indices]

    # Prepare final results
    nms_results = InstanceData()
    nms_results.bboxes = final_bboxes
    nms_results.scores = final_scores
    nms_results.labels = final_labels
    nms_results.keypoints = final_keypoints

    return nms_results


