from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
import torchvision.models as models

from SmArtTorch.utils import *
from SmArtTorch.params import *
from SmArtTorch.layers import *

class Content_Reconstructor(nn.Module):
    #Extract__nn class returns the feature maps of the first 5 conv layers of vgg16.
    def __init__(self, model_path=None):
        super(Content_Reconstructor, self).__init__()
        if model_path == None:
          vgg16 = models.vgg16(pretrained=True).features.eval().to(device)
        else:
          vgg16 = torch.load(model_path).eval().to(device)
        self.layers = list(vgg16.children())
        del vgg16
        self.conv1 = self.layers[0]
        self.conv2 = self.layers[2]
        self.conv3 = self.layers[5]
        self.conv4 = self.layers[7]
        self.conv5 = self.layers[10]
        self.maxpool = self.layers[4]

    def forward(self, crop_content_list):
        self.f_maps = []
        for crop in crop_content_list:
            '''Input is torch tensor with dummy dimension'''
            x = crop.detach().clone()
            out1 = self.conv1(x)
            out1 = F.relu(out1)
            out2 = self.conv2(out1)
            out2 = F.relu(out2)
            out3 = self.maxpool(out2)
            out3 = self.conv3(out3)
            out3 = F.relu(out3)
            out4 = self.conv4(out3)
            out4 = F.relu(out4)
            out5 = self.maxpool(out4)
            out5 = self.conv5(out5)
            out5 = F.relu(out5)
            self.f_maps.append(out4.detach())

    def model_construct(self, layer_count=9):
        '''Construct minimal model for content reconstruction'''
        model = nn.Sequential()
        counter = 0
        for i in range(layer_count):
            layer_name = f'layer_{counter}'
            counter += 1
            model.add_module(layer_name, self.layers[i])
        return model

    def restore(self, crop_stylised_list, epochs, output_freq, lr = 0.0002, verbose=0):
        '''return content-reconstructed image'''
        self.output_imgs = []
        #Using MSE loss
        criterion = nn.MSELoss()
        #Instantiating model with given layers from pretrained VGG16
        model = self.model_construct().to(device)

        #Creating whitenoise image as a starting template
        for crop_stylised, f_map in zip(crop_stylised_list, self.f_maps):
            crop_stylised = crop_stylised.detach().clone()
            img_start = crop_stylised.requires_grad_()

            #Using Adam as an optimiser
            opt = optim.Adam(params= [img_start], lr = lr)

            one_crop_history = []
            for epoch in range(epochs):
                pred = model(img_start)
                loss = criterion(pred, f_map)
                loss.backward()
                opt.step()
                opt.zero_grad()
                if epoch % output_freq == 0:
                    one_crop_history.append(img_start.detach().cpu().data.clamp_(0,1))
                if verbose == 1:
                    if epoch % 20 == 0:
                        print(f'Epoch {epoch}, Loss: {loss}')
                if epoch == epochs-1:
                    one_crop_history.append(img_start.detach().cpu().data.clamp_(0,1))

            self.output_imgs.append(one_crop_history)