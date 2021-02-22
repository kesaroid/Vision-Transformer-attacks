import numpy as np
from torch.autograd import Variable
import torch as torch
import copy
from torch.autograd.gradcheck import zero_gradients

def deepfool(image, model, num_classes=10, overshoot=0.02, max_iter=50, batch_size=64):
    """
       :param image: Image of size HxWx3
       :param model: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """
    is_cuda = torch.cuda.is_available()

    f_image = model.forward(Variable(
        image[:, :, :, :], requires_grad=True)).data.cpu().numpy() # .flatten()

    input_shape = image.cpu().detach().numpy().shape
    pert_image = copy.deepcopy(image)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    x = Variable(pert_image, requires_grad=True)
    fs = model.forward(x)
    
    k_labels = []; pert_images=[]; label_orig = []
    for img_batch in range(image.shape[0]):
        I = (np.array(f_image[img_batch])).flatten().argsort()[::-1]

        # I = I[0:num_classes]
        label = I[0]
        label_orig.append(label)
        loop_i = 0

        fs_list = [fs[img_batch, I[k]] for k in range(num_classes)] # TODO Not needed
        k_i = label

        while k_i == label and loop_i < max_iter:

            pert = np.inf
            fs[img_batch, I[0]].backward(retain_graph=True)
            grad_orig = x.grad.data[img_batch].cpu().numpy().copy()

            for k in range(1, num_classes):
                zero_gradients(x)

                fs[img_batch, I[k]].backward(retain_graph=True)
                cur_grad = x.grad.data[img_batch].cpu().numpy().copy()

                # set new w_k and new f_k
                w_k = cur_grad - grad_orig
                f_k = (fs[img_batch, I[k]] - fs[img_batch, I[0]]).data.cpu().numpy()

                pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

                # determine which w_k to use
                if pert_k < pert:
                    pert = pert_k
                    w = w_k

            # compute r_i and r_tot
            # Added 1e-4 for numerical stability
            r_i = (pert+1e-4) * w / np.linalg.norm(w)
            r_tot = np.float32(r_tot[img_batch] + r_i)

            pert_image[img_batch] = image[img_batch] + (1+overshoot)*torch.from_numpy(r_tot).cuda()
        
            x = Variable(pert_image, requires_grad=True)
            fs = model.forward(x)
            k_i = np.argmax(fs[img_batch].data.cpu().numpy().flatten()) # TODO Don't flatten
            loop_i += 1
        
        k_labels.append(k_i)
        # pert_images.append(pert_image[img_batch])
        r_tot = (1+overshoot)*r_tot

    # pert_images = torch.stack(pert_images, dim=0)
    return r_tot, loop_i, label_orig, k_labels, pert_image