
from __future__ import print_function
import copy
import  csv
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
    
    def forward(self,x):
        return self.model(x)

def test(model, defend=False, output='Results', max_perturb=6.0):
    model.eval()
    fools = 0
    avg_act = 0
    defend_count = 0

    fools_orig = 0
    fools_16x16 = 0

    batch_size = 1
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    iterations = int(100 / batch_size)
    attacker = None

    test_loader = torch.utils.data.DataLoader(torchvision.datasets.CIFAR10(root=ROOT, train=False, 
                            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)]), download=False),
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=4)

    if attack == 'pgd':
        attacker = PGD(model, "cuda:0",
                        norm=pgd_params['norm'],
                        eps=pgd_params['eps'],
                        alpha=pgd_params['alpha'],
                        iters=pgd_params['iterations'],
                        TI=pgd_params['TI'])

    elif attack == 'mifgsm':
        attacker = MIFGSM(model, "cuda:0",
                               eps=mifgsm_params['eps'], 
                               steps=mifgsm_params['steps'], 
                               decay=mifgsm_params['decay'],
                               alpha=mifgsm_params['alpha'],
                               TI=mifgsm_params['TI'],
                               k_=mifgsm_params['k'])

    original_image = []; perturb_image = []
    for i, (data, target) in enumerate(test_loader): # tqdm(test_loader)
        if i == iterations:
            break

        data = data.cuda()
        target = target.cuda()

        # get adversary
        pert_image = get_adversary(model, data, target, attacker)

        if defend:
            # downsize adversary
            pert_16x16 = torch.nn.functional.interpolate(pert_image, size=(16, 16), mode='bilinear', align_corners=False)

        with torch.no_grad():
            # predictions on adversary
            out = torch.nn.Softmax(dim=1).cuda()(model(pert_image))
            act, pred = out.max(1, keepdim=True)

            if defend:
                # predictions on downsized adversary
                defence_out = torch.nn.Softmax(dim=1).cuda()(model(pert_16x16))
                d_act, d_pred = defence_out.max(1,  keepdim=True)

                #assert (pred == d_pred) # This it?

        if defend:
            # defence accuracy calculation (compares predictions original and downsized adversary)
            fools += (pred != target.view_as(pred))[d_pred == pred].sum()

            fools_orig += (pred != target.view_as(pred)).sum()
            fools_16x16 += (d_pred != target.view_as(d_pred)).sum()
            defend_count += (d_pred != pred).sum()
        else:
            # normal accuracy calculation
            fools += (~pred.eq(target.view_as(pred))).sum()

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

        if output:
            if not os.path.exists(output):
                os.mkdir(output)

            # de-normalize
            a = (a * std[0]) + mean[0]
            b = (b * std[0]) + mean[0]

            # save image (this function multiples by 255 and transposes)
            torchvision.utils.save_image(b, os.path.join(output, '{}.png'.format(i+1)))

            # mult by 255 before checking magnitude constraints
            a = a.mul(255).add_(0.5).clamp_(0, 255)
            b = b.mul(255).add_(0.5).clamp_(0, 255)

            sub = b - a

            norm = np.linalg.norm(np.ravel(sub.numpy()), ord=np.inf)
            if norm > max_perturb:
                norms.append(norm)

    if defend:
        print('fool rate with original perturbation: {}'.format(100. * float(fools_orig) / len(test_loader.dataset)))
        print('fool rate with downsized perturbation: {}'.format(100. * float(fools_16x16) / len(test_loader.dataset)))
        print('number of adversaries caught by defense: {}'.format(defend_count))
        print()

    return 100. * float(fools) / len(test_loader.dataset), 100. * float(avg_act) / len(test_loader.dataset), norms

def get_adversary(model, data, target, attacker=None):

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
                                                                    max_iter=deepfool_params['iters'],
                                                                    max_perturb=deepfool_params['eps'],
                                                                    TI=deepfool_params['TI'])
    elif attack == 'pgd':
        pert_image = attacker(data, target)
    elif attack == 'mifgsm':
        pert_image = attacker(data, target)
    else:
        pert_image = data
        
    return pert_image

if __name__ == "__main__":
    # attack = ['sta', 'jacobian', 'carlini', 'lbfgs', 'pixel', 'pgd', 'deepfool', 'mifgsm']
    # Carlini attack takes v long

    max_perturb = 6.0
    iterations = 10
    alpha = 1.0
    TI = True
    k = 5

    if len(sys.argv) > 1:
        attack = sys.argv[1]
    else:
        attack = 'mifgsm'

    defend = True
    pgd_params = {'norm': 'inf', 'eps': max_perturb, 'alpha': alpha, 'iterations': iterations, 'TI': TI}
    deepfool_params = {"iters" : iterations, 'eps': max_perturb, 'TI': TI}
    mifgsm_params = {'eps': max_perturb, 'steps': iterations, 'decay': 1.0, 'alpha': alpha, 'TI': TI, 'k': k}

    model = NN()
    model.cuda()      

    if os.path.isfile("mdl.pth"):
        chk = torch.load("mdl.pth")
        model.load_state_dict(chk["model"])
        del chk
    torch.cuda.empty_cache()

    output_path = None
    if defend:
        output_path = 'Defend_Results_{}'.format(attack)
    else:
        output_path = 'Results_{}'.format(attack)

    acc, _, norms = test(model, defend, output_path, max_perturb)

    print("--------- Running attack: {} with magnitude: {} ---------".format(attack, max_perturb))
    print("--------- Using defense: {} ---------".format(defend))
    print('--------- Fool rate with {} attack: {} ---------'.format(attack, acc))
    print('--------- Perturbation norm that go beyond {}: {} ---------'.format(max_perturb, len(norms)))
    print()
    print(norms)