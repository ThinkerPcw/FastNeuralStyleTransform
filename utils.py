import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import pylab as plt
import numpy as np
import os


def GramMatrix(inputs):
    a, b, c, d = inputs.size()
    features = inputs.view(a, b, c * d)
    G = torch.bmm(features, features.transpose(1, 2))
    return G.div(b * c * d)


def get_image(path, m_trans):
    image = Image.open(path)
    image = m_trans(image)
    image = image.unsqueeze(0)
    return image


def show_image(img, is_show=True):
    img = recover_image(img)
    batch = img.shape[0]
    plt.figure()
    for i in range(batch):
        plt.subplot(1, batch, i + 1)
        plt.imshow(img[i])
    if is_show:
        plt.show()


def save_image(img, path):
    img = recover_image(img)
    img = img[0]
    im = Image.fromarray(img)
    im.save(path)


def recover_image(img):
    return (
            (
                    img.numpy() *
                    np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1)) +
                    np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))
            ).transpose(0, 2, 3, 1) *
            255.
    ).clip(0, 255).astype(np.uint8)


def save_network(path, network, epoch_label, is_only_parameter=True):
    save_filename = 'net_%s.pth' % epoch_label
    save_path = os.path.join(path, save_filename)
    if is_only_parameter:
        state = network.state_dict()
        for key in state: state[key] = state[key].clone().cpu()
        torch.save(network.state_dict(), save_path)
    else:
        torch.save(network.cpu(), save_path)


def restore_network(path, epoch, network=None):
    path = os.path.join(path, 'net_%s.pth' % epoch)
    if network is None:
        network = torch.load(path)
    else:
        network.load_state_dict(torch.load(path))
    if torch.cuda.is_available:
        network.cuda()
    return network
