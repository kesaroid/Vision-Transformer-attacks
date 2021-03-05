from __future__ import print_function
import copy
import csv
import numpy as np
import os
import numpy
import torch
import random
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.nn.init as nninit
from torch.nn.parameter import Parameter
import torchvision.transforms.functional as TF
from vit import VisionTransformer
import cv2

from advertorch import attacks
from deepfool import deepfool
from MIFGSM import MIFGSM
from pgd import PGD
import sys

ROOT = '.'


class NN(nn.Module):
    def __init__(self, img_size=32, patch_size=2):
        super(NN, self).__init__()
        self.model = VisionTransformer(
            img_size=img_size, patch_size=patch_size, in_chans=3, num_classes=10, embed_dim=80, depth=20,
            num_heads=20, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
            drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm)

    def forward(self, x):
        return self.model(x)


def test(model, attack=None, defend=False, output='Results', max_perturb=6.0):
    model.eval()
    correct = 0
    avg_act = 0

    batch_size = 1
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    iterations = int(100 / batch_size)

    test_loader = torch.utils.data.DataLoader(torchvision.datasets.CIFAR10(root=ROOT, train=False,
                                                                           transform=transforms.Compose(
                                                                               [transforms.ToTensor(),
                                                                                transforms.Normalize(mean, std)]),
                                                                           download=True),
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=4)

    if attack == 'pgd':
        pgd_attack = PGD(model, "cuda:0",
                         norm=pgd_params['norm'],
                         eps=pgd_params['eps'],
                         alpha=pgd_params['alpha'],
                         iters=pgd_params['iterations'],
                         TI=pgd_params['TI'])

    elif attack == 'mifgsm':
        mifgsm = MIFGSM(model, loss_fn=nn.CrossEntropyLoss(),
                        mean=mean, std=std,
                        max_norm=max_perturb,
                        norm_type='linf')

    original_image = []
    perturb_image = []
    for i, (data, target) in enumerate(test_loader):  # tqdm(test_loader)
        if i == iterations:
            break

        data = data.cuda()
        target = target.cuda()
        if defend:
            data16x16 = torch.nn.functional.interpolate(data, size=(16, 16), mode='bilinear', align_corners=False)
            data = data16x16

        # TODO Fix epsilon values for all the following attacks
        if attack == 'sta':
            sta = attacks.SpatialTransformAttack(model, num_classes=10)
            pert_image = sta.perturb(data)
        elif attack == 'jacobian':
            jacobian = attacks.JacobianSaliencyMapAttack(model, num_classes=10)
            pert_image = jacobian.perturb(data, target)
        elif attack == 'carlini':
            carlini = attacks.CarliniWagnerL2Attack(model, num_classes=10)
            pert_image = carlini.perturb(data)
        elif attack == 'lbfgs':
            lbfgs = attacks.LBFGSAttack(model, num_classes=10, batch_size=data.shape[0])
            pert_image = lbfgs.perturb(data)
        elif attack == 'pixel':
            pixel = attacks.SinglePixelAttack(model)
            pert_image = pixel.perturb(data)
        elif attack == 'deepfool':
            r, loop_i, label_orig, label_pert, pert_image = deepfool(data.clone().detach().requires_grad_(True),
                                                                     model,
                                                                     max_iter=50,
                                                                     max_perturb=max_perturb)
        elif attack == 'pgd':
            pert_image = pgd_attack(data, target)
        elif attack == 'mifgsm':
            pert_image = mifgsm.attack(data, target)

        with torch.no_grad():
            if not attack:
                pert_image = data
            out = torch.nn.Softmax(dim=1).cuda()(model(pert_image))

        act, pred = out.max(1)

        correct += pred.eq(target.view_as(pred)).sum()
        avg_act += act.sum().data

        original_image.append(data.cpu().detach())
        perturb_image.append(pert_image.cpu().detach())

        assert len(original_image) == len(perturb_image)

    original_image = torch.cat(original_image, dim=0)
    perturb_image = torch.cat(perturb_image, dim=0)

    norms = []
    for i in range(len(original_image)):
        a = original_image[i]
        b = perturb_image[i]

        if not os.path.exists(output):
            os.mkdir(output)

        # de-normalize
        a = (a * std[0]) + mean[0]
        b = (b * std[0]) + mean[0]

        # mult by 255 before checking magnitude constraints
        a = a.mul(255).add_(0.5).clamp_(0, 255)
        b = b.mul(255).add_(0.5).clamp_(0, 255)

        sub = b - a

        norm = np.linalg.norm(np.ravel(sub.numpy()), ord=np.inf)
        if norm > max_perturb:
            norms.append(norm)

    return 100. * float(correct) / len(test_loader.dataset), 100. * float(avg_act) / len(test_loader.dataset), norms


if __name__ == "__main__":
    attack = 'pgd'

    model = NN()
    model.cuda()

    if os.path.isfile("mdl.pth"):
        chk = torch.load("mdl.pth")
        model.load_state_dict(chk["model"])
        del chk

    torch.cuda.empty_cache()

    TIs = [False, True]
    defenses = [False, True]
    max_perturbations = [i for i in range(2, 6)]
    iterations = [i for i in range(1, 11)]

    # run all experiments
    for defend in defenses:
        for max_perturbation in max_perturbations:
            for iteration in iterations:
                for alpha in range(1, max_perturbation):
                    for TI in TIs:

                        print('<----------- PGD with defend={} epsilon={} step={} TI={} iterations={} ----------->'.
                              format(defend, max_perturbation, alpha, TI, iteration), flush=True)

                        pgd_params = {'norm': 'inf',
                                      'eps': max_perturbation,
                                      'alpha': alpha,
                                      'iterations': iteration,
                                      'TI': TI}

                        acc, _, norms = test(model, attack, defend, 'Results_{}'.format(attack), max_perturbation)

                        print('--------- Test accuracy on {} attack: {} ---------'.format(attack, acc))
                        print('--------- Perturbs beyond {}: {} ---------'.format(max_perturbation, len(norms)))
