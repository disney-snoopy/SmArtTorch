import numpy as np
import imageio
import os
import io
from PIL import Image
from torchvision.transforms.functional import resize
from torch.cuda import is_available
from SmArtTorch.utils import loader, unloader
from SmArtTorch.params import device, segmentation_model_path, vgg_model_path
from SmArtTorch.lbfgs_transfer import LBFGS_Transfer
from SmArtTorch.segmentation import Segmentation
from SmArtTorch.content_reconstruction import Content_Reconstructor

class TrainerSegmentation():
    def __init__(self, tensor_content, tensor_style, path_vgg = vgg_model_path, path_seg = segmentation_model_path):
        self.tensor_content = tensor_content.to(device)
        self.tensor_style = tensor_style.to(device)

        # resize depending on gpu availability
        if is_available():
            self.tensor_content_rsz = resize(tensor_content.clone(), 800)
        else:
            self.tensor_content_rsz = resize(tensor_content.clone(), 400)
        self.path_vgg = path_vgg
        self.path_seg = path_seg


    def stylise(self, style_weight = 1e17, epochs = 300, output_freq = 60):
        # instantiate
        self.lbfgs_transfer = LBFGS_Transfer(model_path = self.path_vgg)
        # run style transfer
        self.lbfgs_transfer.learn(content_img=self.tensor_content_rsz,
                                    style_img=self.tensor_style,
                                    input_img=self.tensor_content_rsz,
                                    style_weight=style_weight,
                                    epochs=epochs,
                                    output_freq=output_freq)
        # converting back to original size
        self.lbfgs_transfer.output_imgs = [resize(i, [(self.tensor_content.shape[-2]), (self.tensor_content.shape[-1])]) for i in self.lbfgs_transfer.output_imgs]

        # final stylised image
        self.forward_final = unloader(self.lbfgs_transfer.output_imgs[-1])

    def segmentation(self):
        # runs segmentation and returns cropped images
        self.seg = Segmentation(model_path = self.path_seg)
        self.seg.run_segmentation(unloader(self.tensor_content))

    def seg_crop(self, object_idx):
        self.crop_content, self.crop_style = self.seg.crop_obj(stylised_image = self.forward_final, object_idx = object_idx)

    def content_reconstruction(self, epochs = 300, output_freq = 15, lr = 0.001):

        # instantiating from saved model
        self.cont_recon = Content_Reconstructor(model_path = self.path_vgg)
        # extract feature map from cropped original image
        self.cont_recon.forward(self.crop_content)
        # content reconstruction
        self.cont_recon.restore(crop_stylised_list = self.crop_style,
                                epochs = epochs,
                                output_freq = output_freq,
                                lr = lr)

    def patch(self):
        # patch cropped reconstructed image on stylised image using binary mask
        self.seg.patch(self.cont_recon.output_imgs)
        self.reverse_final = unloader(self.seg.output_recon[-1])

    def generate_gif(self, file_name = 'style_transfer_result.gif', fps = 7):

        images_data_style = [unloader(resize(img, 800)) for img in self.lbfgs_transfer.output_imgs]
        images_data_recon = [unloader(resize(img, 800)) for img in self.seg.output_recon]

        images_data = images_data_style + images_data_recon
        np_imgs = [np.array(img) for img in images_data]
        #extending final picture frames
        for i in range(20):
            np_imgs.append(np_imgs[-1])
        imageio.mimwrite(file_name, np_imgs, fps = fps)