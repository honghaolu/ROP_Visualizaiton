#Step 4: Calculate the IOU of the binary segmentation map during training

import numpy as np
import torch

def calculate_iou(pred,label):
    # Make sure label and pred are PyTorch tensors
    if not isinstance(label, torch.Tensor) or not isinstance(pred, torch.Tensor):
        raise ValueError("label and pred must be PyTorch tensors")

        # Make sure they have the same shape and are on the same device (CPU or GPU)
    if label.shape != pred.shape or label.device != pred.device:
        raise ValueError("label and pred must have the same shape and be on the same device")

        # If the tensor is on the GPU, move it to the CPU first
    if label.device.type == 'cuda':
        label = label.cpu()
        pred = pred.cpu()

    # Binarized prediction results: The prediction results were binarized according to the threshold of 0.5.
    pred_binary = (pred >= 0.5).float()

    # Convert labels (if 255 represents foreground, it is assumed that the labels are already binary)
    label_binary = (label == 1).float()  # Assume that 1 in the label represents the foreground

    # Calculate the intersection (number of pixels that are simultaneously 1)
    intersection = torch.sum(pred_binary * label_binary)

    # Calculate the union (number of pixels that are either 1)
    union = torch.sum(pred_binary) + torch.sum(label_binary) - intersection

    # Avoiding Division by Zero Errors
    if union == 0:
        return 0.0

    # Caculate IoU
    iou = intersection.item() / union.item()  # Use .item() to convert a Tensor to a Python value
    return iou