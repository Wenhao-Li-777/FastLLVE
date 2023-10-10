from typing import Tuple

import torch
from torch.cuda.amp import custom_fwd, custom_bwd

from ._ext import (
    IALUT_cforward, IALUT_cbackward
)


class IALUTTransformFunction(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx,
                video: torch.Tensor,
                lut: torch.Tensor) -> torch.Tensor:

        video = video.contiguous()
        lut = lut.contiguous()

        assert video.ndimension() == 5, \
            "only support 2D image with batch and channel dimensions (5D tensor)"
        assert lut.ndimension() in [6], \
            "only support 4D lookup table with batch dimension (6D tensor)"

        output = video.new_zeros((video.size(0), lut.size(1), video.size(2), video.size(3), video.size(4)))
        IALUT_cforward(video, lut, output)

        ctx.save_for_backward(video, lut)

        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor]:

        grad_output = grad_output.contiguous()

        video, lut = ctx.saved_tensors

        grad_video = torch.zeros_like(video)
        grad_lut = torch.zeros_like(lut)

        IALUT_cbackward(grad_output, video, lut, grad_video, grad_lut)

        return grad_video, grad_lut


def IALUT_transform(
    video: torch.Tensor,
    lut: torch.Tensor) -> torch.Tensor:
    r"""Spatil and Temporal 4D Lookup Table Transform (IALUT-Transform).

    Args:
        video (torch.Tensor): input image of shape (b, 4, t, h, w).
        lut (torch.Tensor): output values of the 4D LUT, shape (b, 3, d, d, d, d).
    Returns:
        torch.Tensor: transformed image of shape (b, 3, t, h, w).
    """
    return IALUTTransformFunction.apply(video, lut)