import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    import scipy.stats as st

    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel

class PGD(nn.Module):
    def __init__(self, model, device, norm, eps, alpha, iters, mean=0.5, std=0.5, TI=False):
        super(PGD, self).__init__()
        assert(2 <= eps <= 10)
        assert(norm in [2, 'inf', np.inf])
        self.eps = (eps / 255.0) / std
        self.alpha = (alpha / 255.0) / std
        self.norm = norm
        self.iterations = iters
        self.loss = nn.CrossEntropyLoss()
        self.model = model
        self.device = device
        self.lower_lim = (0.0 - mean) / std
        self.upper_lim = (1.0 - mean) / std
        self.TI = TI

        kernel = gkern(3, 3).astype(np.float32)
        self.stack_kernel = np.stack([kernel, kernel, kernel])
        self.stack_kernel = np.expand_dims(self.stack_kernel, 0)
        self.stack_kernel = torch.from_numpy(self.stack_kernel).cuda()

    def forward(self, images, labels):
        adv = images.clone().detach().requires_grad_(True).to(self.device)

        for i in range(self.iterations):
            _adv = adv.clone().detach().requires_grad_(True)
            outputs = self.model(_adv)

            self.model.zero_grad()
            cost = self.loss(outputs, labels)
            cost.backward()
            grad = _adv.grad

            if self.TI:
                noise = F.conv2d(grad, self.stack_kernel, stride=1, padding=1)
                noise = noise / torch.mean(torch.abs(noise), [1, 2, 3], keepdim=True)
                grad = grad + noise

            if self.norm in ["inf", np.inf]:
                grad = grad.sign()

            elif self.norm == 2:
                ind = tuple(range(1, len(images.shape)))
                grad = grad / (torch.sqrt(torch.sum(grad * grad, dim=ind, keepdim=True)) + 10e-8)

            assert(images.shape == grad.shape)

            adv = adv + grad * self.alpha

            # project back onto Lp ball
            if self.norm in ["inf", np.inf]:
                adv = torch.max(torch.min(adv, images + self.eps), images - self.eps)

            elif self.norm == 2:
                delta = adv - images

                mask = delta.view(delta.shape[0], -1).norm(self.norm, dim=1) <= self.eps

                scaling_factor = delta.view(delta.shape[0], -1).norm(self.norm, dim=1)
                scaling_factor[mask] = self.eps

                delta *= self.eps / scaling_factor.view(-1, 1, 1, 1)

                adv = images + delta

            adv = adv.clamp(self.lower_lim, self.upper_lim)

        return adv.detach()


