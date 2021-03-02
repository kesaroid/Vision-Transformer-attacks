
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
import cv2

from advertorch import attacks
from deepfool import deepfool
from MIFGSM import MIFGSM
from pgd import PGD

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

def test(model, attack=None, defend=False, output='Results', max_perturb=6):
    model.eval()
    correct = 0
    avg_act = 0

    batch_size = 1
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    iterations = int(10 / batch_size) + 1

    test_loader = torch.utils.data.DataLoader(torchvision.datasets.CIFAR10(root=ROOT, train=False, 
                            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)]), download=True),
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=4
                        )


    if attack == 'pgd':
        pgd_attack = PGD(model, "cuda:0",
                        norm=pgd_params['norm'],
                        eps=pgd_params['eps'],
                        alpha=pgd_params['alpha'],
                        iters=pgd_params['iterations'])
    elif attack == 'mifgsm':
        mifgsm = MIFGSM(model, loss_fn=nn.CrossEntropyLoss(),
                        mean=mean, std=std, 
                        max_norm=6.0)

    original_image = []; perturb_image = []
    for i, (data, target) in enumerate(test_loader): # tqdm(test_loader)
        data = data.cuda()
        target = target.cuda()
        if defend:
            data16x16 = torch.nn.functional.interpolate(data, size=(16, 16),mode='bilinear', align_corners=False)
            data = data16x16
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
            r, loop_i, label_orig, label_pert, pert_image = deepfool(data.clone().detach().requires_grad_(True), model)
        elif attack == 'pgd':
            pert_image = pgd_attack(data, target)
        elif attack == 'mifgsm':
            pert_image = mifgsm.attack(data, target)

        with torch.no_grad():
            if not attack: 
                pert_image = data 
            out = torch.nn.Softmax(dim=1).cuda()(model(pert_image))
                    
        act, pred = out.max(1)
        # Sanity check
        # assert (label_orig == target.detach().cpu().numpy()).all()
        # assert (label_pert == pred.detach().cpu().numpy()).all()
        correct += pred.eq(target.view_as(pred)).sum()
        avg_act += act.sum().data

        original_image.append(data.cpu().detach())
        perturb_image.append(pert_image.cpu().detach())
            
        assert len(original_image) == len(perturb_image)
        if i == iterations: break
    
    original_image = torch.cat(original_image, dim=0)
    perturb_image = torch.cat(perturb_image, dim=0)

    
    norms = []
    for i in range(len(original_image)):
        a = original_image[i]
        b = perturb_image[i]
        
        # sub = torch.subtract(a, b)
        # norms.append(torch.norm(sub)) # p=np.inf

        if output:
            if not os.path.exists(output):
                os.mkdir(output)

            image_a = np.moveaxis(a.clone().cpu().detach().numpy(), 0, -1)
            image_b = np.moveaxis(b.clone().cpu().detach().numpy(), 0, -1)
            # Unnormalize image in order to save
            image_a -= image_a.min(); image_b -= image_b.min() 
            image_a /= image_a.max(); image_b /= image_b.max()
            image_a *= 255; image_b *= 255
            
            # Check perturbations > max value
            sub = np.subtract(image_a, image_b)
            norm = np.linalg.norm(np.ravel(sub), ord=np.inf)
            if norm > max_perturb:
                norms.append(norm)
            cv2.imwrite(os.path.join(output, '{}.jpg'.format(i)), image_b)

    return 100. * float(correct) / len(test_loader.dataset), 100. * float(avg_act) / len(test_loader.dataset), norms


if __name__=="__main__":
    
    # attack = ['sta', 'jacobian', 'carlini', 'lbfgs', 'pixel']
    attack = 'deepfool'
    defend = True
    pgd_params = {'norm': 'inf', 'eps': 6, 'alpha': 1, 'iterations': 20}

    model = NN()
    model.cuda()      

    if os.path.isfile("mdl.pth"):
        chk = torch.load("mdl.pth")
        model.load_state_dict(chk["model"]);
        del chk
    torch.cuda.empty_cache();
    acc,_,norms = test(model, attack, defend)
    

    print('--------- Test accuracy on {} attack: {} ---------'.format(attack, acc))
    print('--------- Perturbation norm that go beyond 6: {} ---------'.format(len(norms)))