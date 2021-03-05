__author__ = 'Kesaroid'

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class MIFGSM():
    def __init__(self, model, loss_fn, mean, std, norm_type='l2', max_norm=6.0, num_iter=10, momentum=0.9, targeted=False):

        super(MIFGSM, self).__init__()
        self.model = model
        self.loss_fn = loss_fn

        mean = np.array(mean)
        std = np.array(std)
        clip_min = (0 - mean) / std
        clip_max = (1 - mean) / std

        channel = mean.shape[0]
        self.mean = torch.Tensor(mean).reshape([channel, 1, 1]).cuda()
        self.std = torch.Tensor(std).reshape([channel, 1, 1]).cuda()
        self.clip_min = torch.Tensor(clip_min).reshape([channel, 1, 1]).cuda()
        self.clip_max = torch.Tensor(clip_max).reshape([channel, 1, 1]).cuda()
        self.max_norm = max_norm
        self.expand_max_norm = torch.Tensor(max_norm / std).cuda()

        self.norm_type = norm_type
        self.num_iter = num_iter
        self.momentum = momentum
        self.targeted = targeted
        self.norm_per_iter = 2 * self.expand_max_norm / self.num_iter

    def attack(self, x, y=None):
        diversity_resize_rate = 1.10
        diversity_prob = 0.0
        delta = self.iterative_gradient_attack(self.model, self.loss_fn, self.targeted,
                                          x, y, self.norm_type, self.max_norm,
                                          self.norm_per_iter, self.num_iter, self.momentum,
                                          diversity_resize_rate, diversity_prob,
                                          self.mean, self.std, self.clip_min, self.clip_max)

        return x + delta

    def iterative_gradient_attack(self, model, loss_fn, targeted, x, y,
                                norm_type, max_norm, max_norm_per_iter,
                                num_iter, momentum,
                                diversity_resize_rate, diversity_prob,
                                mean, std, clip_min, clip_max):
        
        batch_size = x.shape[0]
        delta = torch.zeros_like(x)

        if y is None:
            logits = model(x)
            y = logits.max(1)[1]

        if targeted:
            scaler = -1
        else:
            scaler = 1

        grad = torch.zeros_like(x)

        for i in range(num_iter):
            delta = delta.detach()
            delta.requires_grad = True

            x_diversity = input_pad(x + delta, diversity_resize_rate, diversity_prob=diversity_prob)

            logits = model(x_diversity)
            loss = scaler * loss_fn(logits, y)
            loss.backward()

            noise = delta.grad

            if norm_type == 'l2':
                noise = noise / torch.reshape(torch.norm(torch.reshape(noise, shape=[batch_size, -1]), dim=1), shape=[batch_size, 1, 1, 1])
                grad = grad * momentum + noise
                noise = grad / torch.reshape(torch.norm(torch.reshape(grad, shape=[batch_size, -1]), dim=1), shape=[batch_size, 1, 1, 1])

                # constraint1 : force to satisfy the max norm constaint
                # delta = delta + max_norm_per_iter * noise
                delta = delta + 2 * noise
                # constarint2 : force ot satisfy the image range constaint
                delta = l2_clamp(delta, max_norm)

            elif norm_type == 'linf':
                grad = grad * momentum + noise
                noise = torch.sign(grad)
                # constraint1 : force to satisfy the max norm constaint
                delta = delta.data + max_norm_per_iter * noise
                delta = torch.max(torch.min(delta, max_norm), -max_norm) # if use random init, then need this to enforce satisfy the constraint1
                # constarint2 : force ot satisfy the image range constaint
                delta = torch.max(torch.min(x + delta, clip_max), clip_min) - x

            else:
                raise NotImplementedError('norm_type only can be l1, l2, linf...')

        return delta

def input_pad(x, resize_rate=1.10, diversity_prob=0.3):
    assert resize_rate >= 1.0
    assert diversity_prob >= 0.0 and diversity_prob <= 1.0

    img_size = x.shape[-1]
    img_resize = int(img_size * resize_rate)
    # print(img_size, img_resize, resize_rate)
    rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
    rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
    h_rem = img_resize - rnd
    w_rem = img_resize - rnd
    pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
    pad_right = w_rem - pad_left

    padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

    ret = padded if torch.rand(1) < diversity_prob else x
    return ret

def l2_clamp(delta, max_norm):
    batch_size = delta.shape[0]
    norm = torch.reshape(torch.norm(torch.reshape(delta, shape=[batch_size, -1]), dim=1), shape=[batch_size, 1, 1, 1])
    # if max_norm > norm, mean current l2 norm of delta statisfy the constraint, no need the rescale
    factor = torch.min(max_norm / norm, torch.ones_like(norm))
    delta = delta * factor
    return delta