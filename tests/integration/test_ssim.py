import torch

from splatforge.studio.losses import ssim_loss


def test_ssim_values_and_gradients_are_independent_of_input_image_layout():
    torch.manual_seed(150)
    prediction = torch.rand(17, 19, 3, dtype=torch.float64, requires_grad=True)
    target = torch.rand_like(prediction)
    chw_storage = (
        prediction.detach()
        .permute(2, 0, 1)
        .contiguous()
        .permute(1, 2, 0)
        .requires_grad_()
    )
    first = ssim_loss(prediction, target)
    second = ssim_loss(chw_storage, target)
    torch.testing.assert_close(first, second)
    a = torch.autograd.grad(first, prediction)[0]
    b = torch.autograd.grad(second, chw_storage)[0]
    torch.testing.assert_close(a, b)
    direction = torch.randn_like(prediction)
    eps = 1e-5
    numerical = (
        ssim_loss(prediction.detach() + eps * direction, target)
        - ssim_loss(prediction.detach() - eps * direction, target)
    ) / (2 * eps)
    torch.testing.assert_close((a * direction).sum(), numerical, atol=1e-8, rtol=1e-5)
