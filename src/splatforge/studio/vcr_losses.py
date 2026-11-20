"""Normal, distortion and image supervision for VCR-GauS."""

import torch
import torch.nn.functional as F


def normal_l1_cosine(prediction, target, weight=None):
    """MonoSDF-style normal loss: component L1 plus one minus cosine."""
    if weight is None:
        weight = torch.ones_like(prediction[..., 0])
    weight = weight.to(prediction.dtype)
    l1 = (weight * (prediction - target).abs().sum(-1)).mean()
    cosine = (weight * (1.0 - (prediction * target).sum(-1))).mean()
    return l1 + cosine


def cosine_confidence(rendered_normal, prior_normal, temperature=0.01):
    """Confidence that the rendered normal agrees with the prior normal."""
    cosine = (rendered_normal * prior_normal).sum(-1).clamp(-1.0, 1.0)
    if temperature > 0:
        return torch.exp((cosine - 1.0) / temperature).detach()
    return torch.ones_like(cosine)


def normal_curvature(normal):
    """Discrete curvature of a normal map (VCR-GauS normal2curv)."""
    padded = F.pad(
        normal.permute(2, 0, 1)[None], (1, 1, 1, 1), mode="replicate"
    )
    center = padded[:, :, 1:-1, 1:-1]
    up = padded[:, :, :-2, 1:-1] - center
    left = padded[:, :, 1:-1, :-2] - center
    down = padded[:, :, 2:, 1:-1] - center
    right = padded[:, :, 1:-1, 2:] - center
    curvature = (up + left + down + right)[0].permute(1, 2, 0)
    return curvature.norm(dim=-1)


def edge_aware_weight(image):
    """Image-gradient weighting used by the VCR-GauS distortion loss.

    The reference implementation weights the distortion map by the exponential
    of the strongest local colour gradient, so that texture edges do not pull
    the surface.
    """
    if min(image.shape[:2]) < 3:
        return image.new_zeros(image.shape[:2])
    channels_first = image.permute(2, 0, 1)
    grad_left = torch.mean(
        torch.abs(channels_first[:, 1:-1, 1:-1] - channels_first[:, 1:-1, :-2]), 0
    )
    grad_right = torch.mean(
        torch.abs(channels_first[:, 1:-1, 1:-1] - channels_first[:, 1:-1, 2:]), 0
    )
    grad_top = torch.mean(
        torch.abs(channels_first[:, 1:-1, 1:-1] - channels_first[:, :-2, 1:-1]), 0
    )
    grad_bottom = torch.mean(
        torch.abs(channels_first[:, 1:-1, 1:-1] - channels_first[:, 2:, 1:-1]), 0
    )
    strongest = torch.max(
        torch.stack((grad_left, grad_right, grad_top, grad_bottom), dim=-1), dim=-1
    )[0]
    weight = torch.exp(-strongest)
    return F.pad(weight, (1, 1, 1, 1), mode="constant", value=0)


def gaussian_window(window_size: int, channels: int, device, dtype):
    """The reference 11x11 gaussian window of the 3DGS SSIM."""
    sigma = 1.5
    coordinates = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    gaussian = torch.exp(-(coordinates ** 2) / (2 * sigma ** 2))
    gaussian = (gaussian / gaussian.sum()).unsqueeze(1)
    window = gaussian @ gaussian.t()
    return window.expand(channels, 1, window_size, window_size).contiguous()


def reference_ssim_loss(prediction, target, window_size=11):
    """SSIM with the reference gaussian window.

    The studio SSIM uses a uniform window as a fast approximation; the
    reference implementation uses a gaussian window, which changes both the
    loss value and its gradient.
    """
    first = prediction.permute(2, 0, 1)[None]
    second = target.permute(2, 0, 1)[None]
    channels = first.shape[1]
    window = gaussian_window(
        window_size, channels, first.device, first.dtype
    )
    padding = window_size // 2
    mu_first = F.conv2d(first, window, padding=padding, groups=channels)
    mu_second = F.conv2d(second, window, padding=padding, groups=channels)
    mu_first_sq = mu_first.pow(2)
    mu_second_sq = mu_second.pow(2)
    mu_product = mu_first * mu_second
    sigma_first_sq = (
        F.conv2d(first * first, window, padding=padding, groups=channels)
        - mu_first_sq
    )
    sigma_second_sq = (
        F.conv2d(second * second, window, padding=padding, groups=channels)
        - mu_second_sq
    )
    sigma_product = (
        F.conv2d(first * second, window, padding=padding, groups=channels)
        - mu_product
    )
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    score = ((2 * mu_product + c1) * (2 * sigma_product + c2)) / (
        (mu_first_sq + mu_second_sq + c1)
        * (sigma_first_sq + sigma_second_sq + c2)
    )
    return 1.0 - score.mean()
