import streamlit as st
from PIL import Image
import torch
import matplotlib.pyplot as plt
from torch import nn
from models import Generator
import numpy as np
from torchvision import transforms
from torchvision.utils import make_grid
from dotenv import load_dotenv
import os

load_dotenv()

DEVICE = os.getenv('DEVICE')


class CNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(kernel_size=5, in_channels=1, out_channels=16)
        self.conv2 = nn.Conv2d(kernel_size=5, in_channels=16, out_channels=32)

        self.fc = nn.Linear(in_features=512, out_features=10)

        self.maxpool = nn.MaxPool2d(kernel_size=2)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


def recognition_digit(pil_img):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,)),
                                    ])
    cnnet = torch.load("cnnet").to('cpu')
    pil_img = pil_img.resize((28, 28)).convert('1')
    img_transformed = transform(pil_img)
    img_transformed = img_transformed.unsqueeze(0)
    with torch.no_grad():
        output = cnnet(img_transformed.to(torch.float32))
    _, predicted = torch.max(output.data, 1)
    return predicted.item()


def get_distrib():
    model_epoch = 49

    model = Generator(2).to(DEVICE)
    load_checkpoint(
        checkpoint=torch.load(f'weights/checkpoint_{model_epoch}.pth.tar'), model=model)
    model.eval()

    res = []

    r1, r2 = -10, 10

    for i in range(20000):
        a, b = (r1 - r2) * torch.rand(2) + r2
        coord = torch.tensor([a, b]).to(DEVICE)
        fake = model(coord.unsqueeze(0)).detach()
        fake = (fake + 1) / 2
        fake = fake[0][0].cpu().numpy()
        digit = recognition_digit(Image.fromarray(np.uint8(fake * 255)))
        res.append([a.item(), b.item(), digit])

    res = np.array(res)
    fig = plt.figure(figsize=(10, 10))
    plt.scatter(res[:, 0], res[:, 1], c=res[:, 2], cmap='tab10')
    plt.colorbar()
    plt.xlim(r1, r2)
    plt.ylim(r1, r2)
    plt.show()


def save_checkpoint(state, epoch):
    print("=> Saving checkpoints")
    filename = f'weights/checkpoint_{epoch}.pth.tar'
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    model.load_state_dict(checkpoint)


if __name__ == '__main__':
    get_distrib()
