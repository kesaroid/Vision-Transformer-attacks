__author__ = 'Kesaroid'

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class MIFGSM(nn.Module):
    """
    MI-FGSM from the paper 'Boosting Adversarial Attacks with Momentum'
    https://arxiv.org/abs/1710.06081

    # Only Linf implemented

    Arguments:
        model: model to attack.
        eps: maximum perturbation
        decay: momentum factor
        steps: number of iterations

    """

    def __init__(self, model, device, eps=6.0, steps=5, decay=1.0, mean=0.5, std=0.5):
        super(MIFGSM, self).__init__()
        self.model = model
        self.eps = (eps / 255.0) / std
        self.steps = steps
        self.decay = decay
        self.device = device
        self._targeted = -1
        self.alpha = self.eps / self.steps

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        # labels = self._transform_label(images, labels)
        
        loss = nn.CrossEntropyLoss()
        momentum = torch.zeros_like(images).detach().to(self.device)

        adv_images = images.clone().detach()
        
        for i in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)

            cost = self._targeted*loss(outputs, labels)
            
            grad = torch.autograd.grad(cost, adv_images, 
                                       retain_graph=False, create_graph=False)[0]
            
            grad_norm = torch.norm(nn.Flatten()(grad), p=1, dim=1)
            grad = grad / grad_norm.view([-1]+[1]*(len(grad.shape)-1))
            grad = grad + momentum*self.decay
            momentum = grad

            adv_images = adv_images.detach() - self.alpha*grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            
            adv_images = torch.max(torch.min(adv_images, images + self.eps), images - self.eps)

        return adv_images