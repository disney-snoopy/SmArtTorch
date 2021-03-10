import streamlit as st
from PIL import Image
import time
import copy
import io
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torch
import base64
from SmArtTorch.utils import loader, unloader
from SmArtTorch.trainer_segmentation import TrainerSegmentation
from SmArtTorch.params import *

# setting wide screen
st.set_page_config(layout="wide")

# title
st.title("SmArt Generative Service")
if torch.cuda.is_available():
    st.sidebar.write('This app is powered by GPU')
else:
    st.sidebar.write('This app is powered by CPU')

st.subheader('SmArt Generative User Guide')
'''
1. Upload your image to styise. \n
2. You can either upload your style image or choose from our curated gallery. \n
3. Choose the people you want to restore! \n
4. Also checkout the gif file!
'''

# multi column layout and headers
c1, c2 = st.beta_columns((1, 1))
c1.header("Content picture")
c2.header("Style picture")
st.subheader('Style Gallery')

# hyperparams
style_weight = 1e15

# transfer strength side bar
transfer_strength_slider = st.sidebar.select_slider(label = 'Style Transfer Strength',
                                                    options = ['Test', 'Weak', 'Average', 'Strong'],
                                                    value = 'Average')
transfer_epoch_dict = {'Test':(10, 2),
                       'Weak':(150, 15),
                       'Average':(250, 25),
                       'Strong':(400, 40)}

transfer_epoch = transfer_epoch_dict[transfer_strength_slider]

# restore strength side bar
restore_strength_slider = st.sidebar.select_slider(label = 'Restore Strength',
                                                    options = ['Test', 'Weak', 'Average', 'Strong'],
                                                    value = 'Average')
restore_epoch_dict = {'Test':(10, 2),
                       'Weak':(100,10),
                       'Average':(200, 20),
                       'Strong':(300, 30)}

restore_epoch = restore_epoch_dict[restore_strength_slider]

# Descrition
st.sidebar.write('Stronger style transfer and restoration take longer time!')
st.sidebar.write('If your session is powered by CPU, consider using lower strength')

# whether to export gif
check_gif = st.sidebar.checkbox('Export transformation gif')

def main():
    #dummy variables for if statements
    forward_final = None
    fig = None
    crop_boolean = None
    num_objects = None
    run_restoration = None
    object_idx = None


    # img upload
    c2.subheader('You can upload your own style picture or choose from our gallery below')
    content_up = c1.file_uploader("Upload your picture for style transfer", type=['png', 'jpg', 'jpeg'])
    style_up = c2.file_uploader("Upload your favourite style picture", type=['png', 'jpg', 'jpeg'])
    style_choose = c2.multiselect('Choose from Gallery', range(1,22))
    if len(style_choose) != 0:
        style_up = gallery_path[style_choose[0]-1]

    # getting gallery images
    st.image('https://res.cloudinary.com/dbxctsqiw/image/upload/v1615307946/gallery_1_jdruxe.png', use_column_width = True)
    st.image('https://res.cloudinary.com/dbxctsqiw/image/upload/v1615307945/gallery_2_zyijkf.png', use_column_width = True)


    # Once content picture is uploaded, resize and display
    if content_up is not None:
        img_content = Image.open(content_up)
        c1.image(img_content, caption='Image to Stylise.', use_column_width=True)

    # Display style image once uploaded
    if style_up is not None:
        img_style = Image.open(style_up)
        c2.image(img_style, caption='Your Style Image.', use_column_width=True)

    # get trainer class
    if content_up is not None and style_up is not None:
        time.sleep(0.5)
        forward_final = get_stylise(image_content = content_up, image_style = style_up,
                            style_weight = style_weight, epochs = transfer_epoch, output_freq = 3)
        st.subheader('Your stylised picture!')
        st.image(forward_final, use_column_width = True)

    if content_up is not None and style_up is not None and forward_final is not None:
        time.sleep(0.5)
        img, num_objects = get_segmentation(image_content = content_up, image_style = style_up)
        st.subheader('Choose the objects you want to restore from the sidebar!')
        st.image(img)
        object_idx = st.sidebar.multiselect('Choose the objects numbers you want to maintain', range(num_objects))

    if object_idx is not None:
        btn_restore = st.sidebar.button('run restoration')
        if btn_restore:
            if len(object_idx) == 0:
                st.sidebar.write('Choose object number from the dropdown!')
            else:
                reverse_final = get_restoration(image_content = content_up, image_style = style_up,
                                                style_weight = style_weight, epochs = restore_epoch,
                                                output_freq = 3, object_idx = object_idx, check_gif = check_gif)
                st.subheader('Restored style transfer!')
                st.image(reverse_final, use_column_width = True)

                if check_gif:
                    # display gif
                    st.subheader('This is your style transfer gif!')
                    st.image("style_transfer_result.gif", use_column_width = True)

def hash_func(obj):
    return [obj.detach().cpu().numpy()]

@st.cache(hash_funcs={torch.Tensor: hash_func})
def get_stylise(image_content, image_style, style_weight, epochs, output_freq):
    tensor_content = loader(image_content)
    tensor_style = loader(image_style)
    trainer = TrainerSegmentation(tensor_content=tensor_content,
                                tensor_style=tensor_style,
                                path_vgg=vgg_model_path,
                                path_seg=segmentation_model_path)
    trainer.stylise(style_weight = style_weight, epochs = transfer_epoch[0], output_freq = transfer_epoch[0])
    return trainer.forward_final

@st.cache(hash_funcs={torch.Tensor: hash_func})
def get_segmentation(image_content, image_style):
    tensor_content = loader(image_content)
    tensor_style = loader(image_style)

    trainer = TrainerSegmentation(tensor_content=tensor_content,
                                tensor_style=tensor_style,
                                path_vgg=vgg_model_path,
                                path_seg=segmentation_model_path)
    trainer.segmentation()
    fig, num_objects = trainer.seg.plot_box_ind(threshold = 0.4)
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img, num_objects

@st.cache(hash_funcs={torch.Tensor: hash_func})
def get_restoration(image_content, image_style, style_weight, epochs, output_freq, object_idx, check_gif):
    tensor_content = loader(image_content)
    tensor_style = loader(image_style)
    trainer = TrainerSegmentation(tensor_content=tensor_content,
                                tensor_style=tensor_style,
                                path_vgg=vgg_model_path,
                                path_seg=segmentation_model_path)
    trainer.segmentation()
    trainer.stylise(style_weight = style_weight, epochs = transfer_epoch[0], output_freq = transfer_epoch[1])
    trainer.seg_crop(object_idx = object_idx)
    trainer.content_reconstruction(lr = 0.001, epochs = restore_epoch[0], output_freq=restore_epoch[1])
    trainer.patch()
    reverse_final = trainer.reverse_final
    if check_gif:
        gif = trainer.generate_gif(fps = 5)
    return reverse_final


if __name__ == "__main__":
    main()