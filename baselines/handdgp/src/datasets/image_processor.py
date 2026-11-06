from typing import Optional
import numpy as np
import torch


class ImageProcessor:

    @staticmethod
    def preprocess(image: np.ndarray, input_size: Optional[int] = None) -> torch.Tensor:
        """
        Convert an image into a tensor with the right shape for model input.

        Args:
            image (np.ndarray): The input image as a NumPy array.
            input_size (Optional[int]): If specified, resize the image to (input_size, input_size).

        Returns:
            torch.Tensor: The preprocessed image tensor.
        """
        image_ = torch.from_numpy(image).float() / 255.0
        image_ = image_.permute(2, 0, 1).unsqueeze(0)
        if input_size is not None:
            image_ = torch.nn.functional.interpolate(
                image_,
                size=(input_size, input_size),
                mode="bilinear",
                align_corners=False,
            )
        return image_.squeeze(0)
