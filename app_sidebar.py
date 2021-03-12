import streamlit as st
from PIL import Image
import requests
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

# creator expander
expander_creators = st.beta_expander("👨‍🦰👨‍🦱Creators of SmArt👩‍🦱👱‍♂️", expanded=False)
with expander_creators:
    a1, a2, a3, a4, a5 = st.beta_columns((1,1,1,1,1))
    a1.image('https://res.cloudinary.com/dbxctsqiw/image/upload/v1615466629/SmArt/Jae_s0lcs8.png')
    a2.image('https://res.cloudinary.com/dbxctsqiw/image/upload/v1615466627/SmArt/ed_ljtaqb.png')
    a3.image('https://res.cloudinary.com/dbxctsqiw/image/upload/v1615466627/SmArt/peter_qleb71.png')
    a4.image('https://res.cloudinary.com/dbxctsqiw/image/upload/v1615466629/SmArt/omer_gjio1o.png')
    a5.image('https://res.cloudinary.com/dbxctsqiw/image/upload/v1615482783/SmArt/julio_ejdzks.png')
    p1, p2, p3, p4, p5 = st.beta_columns((1,1,1,1,1))
    p1.markdown('''**Jae Kim**''')
    p1.markdown('''byungjae91@gmail.com''')
    p2.markdown('''**Edward Touche**''')
    p2.markdown('''edwardtouche@gmail.com''')
    p3.markdown('''**Peter Stanley**''')
    p3.markdown('''peterstanley1@live.com''')
    p4.markdown('''**Omer Aziz**''')
    p4.markdown('''omeraziz10@gmail.com''')
    p5.markdown('''**Julio Quintana**''')

# title
st.title("SmArt Generative Service")

# user guide expander
st.markdown('''---''')
expander_user_guide = st.beta_expander("📕User Guide", expanded=False)
with expander_user_guide:
    clicked = st.markdown("""
        #### 1. 🍊Upload your image to stylise.
        We support png, jpg and jpeg formats.

        #### 2. 🎨Upload a picture to take style from.
        Or you can choose from our curated style gallery.

        #### 3. 🏃‍♂️Run SmArt!
        SmArt will start working on your pictures automatically after you upload both content and style pictures.
        The process can take up to two minutes.
        Your stylised picture will be displayed at the bottom of this webpage.

        #### 4. 🕵️‍♂️Lost some details you wanted to keep?
        Sometimes SmArt blurs away details that you want to keep.
        Our algorithm recognises human contours.
        You will be able choose which people's details you want to restore after reviewing your stylised picture.

        #### 5. 🎥Want some gif?
        Don't forget to check gif files showing the transformation process.
        gif will be provided after the stylising is completed.

        #### 6. 🧮Changing the settings.
        You can control the output images by adjusting style and restoration strengths from the left sidebar.
        It is recommended to use the default setting first and adjust the strengths to generate images that suit your preferences.
        """)
st.markdown('''___''')

# multi column layout and headers
c1, c2 = st.beta_columns((1, 1))
c1.subheader("""**1. Content picture🍊**""")
c2.subheader("""**2. Style picture🎨**""")

##############################Sidebar config###############################################
# sidebar title
st.sidebar.subheader('SmArt Control Panel')

# hyperparams
style_weight = 1e6

# transfer strength side bar
transfer_strength_slider = st.sidebar.select_slider(label = 'Style Strength',
                                                    options = ['Test', 'Weak', 'Average', 'Strong'],
                                                    value = 'Average')
transfer_epoch_dict = {'Test':(2, 2),
                       'Weak':(50, 4),
                       'Average':(100, 6),
                       'Strong':(250, 10)}

transfer_epoch = transfer_epoch_dict[transfer_strength_slider]

# restore strength side bar
restore_strength_slider = st.sidebar.select_slider(label = 'Restoration Strength',
                                                    options = ['Test', 'Weak', 'Average', 'Strong'],
                                                    value = 'Average')
restore_epoch_dict = {'Test':(2, 2),
                       'Weak':(100,4),
                       'Average':(150, 5),
                       'Strong':(250, 7)}

restore_epoch = restore_epoch_dict[restore_strength_slider]

# Display backend type
if torch.cuda.is_available():
    st.sidebar.markdown('''⚡ This app is currently powered by **GPU**.''')
else:
    st.sidebar.markdown('''⚠️ This app is currently powered by **CPU**.''')
    st.sidebar.write('⏰ Stronger style transfer and restoration take longer time!\
                          If your session is powered by CPU, consider using lower strengths.')


##############################Sidebar config###############################################


def main():
    #dummy variables for if statements
    forward_final = None
    fig = None
    crop_boolean = None
    num_objects = None
    run_restoration = None
    object_idx = None


    # img upload
    content_up = c1.file_uploader("Upload your picture for style transfer", type=['png', 'jpg', 'jpeg'])
    style_up = c2.file_uploader("Upload your favourite style picture", type=['png', 'jpg', 'jpeg'])
    style_choose = c2.multiselect('Alternatively, choose from Gallery. Expand the gallery below.', range(1,22))



    #gallery_path = '/content/drive/MyDrive/Lewagon/project/style_transfer/deployment_colab/gallery_choices'
    if len(style_choose) != 0:
        file_name = gallery_path[style_choose[-1]-1]
        style_up = file_name

    # getting gallery images
    st.markdown('''---''')
    expander_gallery = st.beta_expander("👉 Click here to see our style gallery", expanded=False)
    with expander_gallery:
        st.image('https://res.cloudinary.com/dbxctsqiw/image/upload/v1615307946/gallery_1_jdruxe.png', use_column_width = True)
        st.image('https://res.cloudinary.com/dbxctsqiw/image/upload/v1615307945/gallery_2_zyijkf.png', use_column_width = True)

    if content_up is not None or style_up is not None:
        q1, q2 = st.beta_columns((9, 1))
        q1.markdown('''___''')
        q1.subheader('☑️Pictures of your choice')
        d1, d2 = st.beta_columns((1, 1))
        d1.write('🍊Content')
        d2.write('🎨Style')

    # Once content picture is uploaded, resize and display
    if content_up is not None:
        img_content = Image.open(content_up)
        d1.image(img_content, use_column_width=True)

    # Display style image once uploaded
    if style_up is not None:
        img_style = Image.open(style_up)
        d2.image(img_style, use_column_width=True)

    if content_up is not None and style_up is not None:
        f1, f2 = st.beta_columns((9, 1))
        f1.markdown('''___''')
        f1.subheader('🖌️Your stylised picture!')

    e1, e2 = st.beta_columns((1, 1))
    # get trainer class
    if content_up is not None and style_up is not None:
        time.sleep(0.5)
        if forward_final is None:
            forward_final = get_stylise(image_content = content_up, image_style = style_up,
                                style_weight = style_weight, epochs = transfer_epoch[0], output_freq = transfer_epoch[1])
            e1.image(forward_final, use_column_width = True)
            e2.markdown('''&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;👈 Expand!''')


    if content_up is not None and style_up is not None and forward_final is not None:
        time.sleep(0.5)
        img, num_objects = get_segmentation(image_content = content_up, image_style = style_up)
        if num_objects == 0:
          # if no human objects are found, ask whether user wants gif
            e1.write('🕵️‍♂️We could not find any human objects in your picture!')
            gif_button = e1.button('Get me gif!')
            if gif_button:
                t1, t2 = st.beta_columns((9, 1))
                t1.markdown('''___''')
                t1.subheader('🎥Enjoy your gif!')
                y1, y2 = st.beta_columns((1, 1))
                y1.image("style_transfer_result.gif", use_column_width = True, output_format = 'GIF')
                y2.markdown('''&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;👈 Expand!''')
                y2.markdown('''🚩Enjoy your stylised picture.''')
                y2.markdown('''🙈It can take some time to fully load the gif. It will animate smoothly once fully loaded.''')
                y2.markdown('''🎮 Why don\'t you play with different style strengths? Check out the sidebar!''')
                y2.markdown('''⚠️Current stylised picture will be lost as soon as you change the settings.''')
                y2.markdown('''💾Please save now to keep the pictures''')
                y2.markdown('''🔄To start over with new images, refresh the page or hit F5''')

        else:
            u1, u2 = st.beta_columns((9, 1))
            u1.markdown('''___''')
            u1.subheader('🕵️‍♂️Choose the people you want to restore!')
            g1, g2 = st.beta_columns((2, 1))
            g1.image(img)
            g2.text('')
            g2.text('')
            g2.text('')
            g2.text('')
            g2.text('')
            g2.write('Enter your choice here')
            object_idx = g2.multiselect('Choose the objects numbers you want to maintain', range(num_objects))

    if object_idx is not None:
        btn_restore = g2.button('run restoration')
        if btn_restore:
            if len(object_idx) == 0:
                g2.write('Choose object number from the dropdown!')
            else:
                reverse_final = get_restoration(image_content = content_up, image_style = style_up,
                                                style_weight = style_weight, epochs = restore_epoch[0],
                                                output_freq = restore_epoch[1], object_idx = object_idx)
                u1, u2 = st.beta_columns((9, 1))
                u1.markdown('''___''')
                u1.subheader('🧑‍🔧You details are restored!')
                i1, i2 = st.beta_columns((1, 1))
                i1.image(reverse_final, use_column_width = True)
                i2.markdown('''👇 Don't forget to scroll down to see your gif!''')

                o1, o2 = st.beta_columns((9, 1))
                o1.markdown('''___''')
                o1.subheader('🎥Enjoy your gif!')
                r1, r2 = st.beta_columns((1, 1))
                r1.image("style_transfer_result.gif", use_column_width = True, output_format = 'PNG')
                r2.markdown('''&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;👈 Expand!''')
                r2.markdown('''🚩Enjoy your stylised picture.''')
                r2.markdown('''🙈It can take some time to fully load the gif. It will animate smoothly once fully loaded.''')
                r2.markdown('''🎮 Why don\'t you play with different style strengths? Check out the sidebar!''')
                r2.markdown('''⚠️Current stylised picture will be lost as soon as you change the settings.''')
                r2.markdown('''💾Please save now to keep the pictures''')
                r2.markdown('''🔄To start over with new images, refresh the page or hit F5''')

        btn_gif_2 = g2.button('Get me gif without restoration')
        if btn_gif_2:
            w1, w2 = st.beta_columns((9, 1))
            w1.markdown('''___''')
            w1.subheader('🎥Enjoy your gif!')
            r1, r2 = st.beta_columns((1, 1))
            r1.image("style_transfer_result.gif", use_column_width = True, output_format = 'PNG')
            r2.markdown('''&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;👈 Expand!''')
            r2.markdown('''🚩Enjoy your stylised picture.''')
            r2.markdown('''🙈It can take some time to fully load the gif. It will animate smoothly once fully loaded.''')
            r2.markdown('''🎮 Why don\'t you play with different style strengths? Check out the sidebar!''')
            r2.markdown('''⚠️Current stylised picture will be lost as soon as you change the settings.''')
            r2.markdown('''💾Please save now to keep the pictures''')
            r2.markdown('''🔄To start over with new images, refresh the page or hit F5''')


def hash_func(obj):
    return [obj.detach().cpu().numpy()]

@st.cache(hash_funcs={torch.Tensor: hash_func})
def get_stylise(image_content, image_style, style_weight, epochs , output_freq):

    image_content=Image.open(image_content).convert('RGB')
    tensor_content = T.ToTensor()(image_content).unsqueeze(0).to(device)

    image_style=Image.open(image_style).convert('RGB')
    tensor_style = T.ToTensor()(image_style).unsqueeze(0).to(device)

    trainer = TrainerSegmentation(tensor_content=tensor_content,
                                tensor_style=tensor_style,
                                path_vgg=vgg_model_path,
                                path_seg=segmentation_model_path)
    trainer.stylise(style_weight = style_weight, epochs = epochs, output_freq = output_freq)
    # dummy class
    class dummy:
      a = None
    trainer.seg = dummy
    trainer.seg.output_recon = []
    gif = trainer.generate_gif(fps = 10)
    return trainer.forward_final

@st.cache(hash_funcs={torch.Tensor: hash_func})
def get_segmentation(image_content, image_style):
    image_content=Image.open(image_content).convert('RGB')
    tensor_content = T.ToTensor()(image_content).unsqueeze(0).to(device)

    image_style=Image.open(image_style).convert('RGB')
    tensor_style = T.ToTensor()(image_style).unsqueeze(0).to(device)

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
def get_restoration(image_content, image_style, style_weight, epochs, output_freq, object_idx):
    image_content=Image.open(image_content).convert('RGB')
    tensor_content = T.ToTensor()(image_content).unsqueeze(0).to(device)

    image_style=Image.open(image_style).convert('RGB')
    tensor_style = T.ToTensor()(image_style).unsqueeze(0).to(device)

    trainer = TrainerSegmentation(tensor_content=tensor_content,
                                tensor_style=tensor_style,
                                path_vgg=vgg_model_path,
                                path_seg=segmentation_model_path)
    trainer.segmentation()
    trainer.stylise(style_weight = style_weight, epochs = transfer_epoch[0], output_freq = transfer_epoch[1])
    trainer.seg_crop(object_idx = object_idx)
    trainer.content_reconstruction(lr = 0.0015, epochs = restore_epoch[0], output_freq = restore_epoch[1])
    trainer.patch()
    reverse_final = trainer.reverse_final
    gif = trainer.generate_gif(fps = 10)
    return reverse_final

if __name__ == "__main__":
    main()