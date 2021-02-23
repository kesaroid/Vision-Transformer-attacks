
from __future__ import print_function
import copy
import  csv
import numpy as np
import os
import numpy
import torch
import random
# from tqdm import tqdm
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

ROOT = '.'

test_loader = torch.utils.data.DataLoader(torchvision.datasets.CIFAR10(root=ROOT, train=False, transform=transforms.Compose([
                        transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]), download=True),
                        batch_size=2,
                        shuffle=False,
                        num_workers=4
                        )

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.model = VisionTransformer(
        img_size=32, patch_size=2, in_chans=3, num_classes=10, embed_dim=80, depth=20,
                 num_heads=20, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm)
    
    def forward(self,x):
        return self.model(x)

def test(model, test_loader, attack):
    model.eval()
    correct = 0
    avg_act = 0
    for data, target in test_loader:
        data = data.cuda()
        target = target.cuda()
        
        if attack == 'sta':
            sta = attacks.SpatialTransformAttack(model, num_classes=10)
            pert_image = sta.perturb(data)
        elif attack == 'deepfool':
            r, loop_i, label_orig, label_pert, pert_image = deepfool(torch.tensor(data,requires_grad =True), model)

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

    return 100. * float(correct) / len(test_loader.dataset),100. * float(avg_act) / len(test_loader.dataset)


if __name__=="__main__":
        model = NN()
        model.cuda()

        attack = 'deepfool'

        if os.path.isfile("mdl.pth"):
            chk = torch.load("mdl.pth")
            model.load_state_dict(chk["model"]);
            del chk
        torch.cuda.empty_cache();
        acc,_ = test(model,test_loader, attack)
        

        print('--------- Test accuracy on {} attack: {} ---------'.format(attack, acc))