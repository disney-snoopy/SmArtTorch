import torch
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vgg_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
vgg_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)


# home path
home = os.getcwd()



# model paths for streamlit
# need to be updated for gcp
vgg_model_path = os.path.join(home, 'pretrained_models/vgg16_pretrained')
segmentation_model_path = os.path.join(home, 'pretrained_models/torch_segmentation_finetuned')

# gallery wall paper
gallery_1 = os.path.join(home, 'images/gallery_1.png')
gallery_2 = os.path.join(home, 'images/gallery_2.png')


# gallery images paths
path_1 = os.path.join(home, 'images/Leonid_Afremov-AlleyByTheLake.jpg')
path_2 = os.path.join(home, 'images/adam-styka-youngwaterbearer.jpeg')
path_3 = os.path.join(home, 'images/david-michael-hinnebusch-sleep-til-spring.jpg')
path_4 = os.path.join(home, 'images/GeorgesSeurat-sunday_afternoon.jpg')
path_5 = os.path.join(home, 'images/houria-niati-beaute-calme-et-volupte.jpg')

path_6 = os.path.join(home, 'images/hands_flowers_eyes.jpg')
path_7 = os.path.join(home, 'images/kandinsky_street.jpg')
path_8 = os.path.join(home, 'images/JacksonPollock-PoliticalConvergence.jpg')
path_9 = os.path.join(home, 'images/mona_lisa.jpg')
path_10 = os.path.join(home, 'images/munch_scream.jpg')

path_11 = os.path.join(home, 'images/michelangelo-creation-of-adam.jpg')
path_12 = os.path.join(home, 'images/van-gogh-wheatfields.jpeg')
path_13 = os.path.join(home, 'images/starry_night.jpg')
path_14 = os.path.join(home, 'images/KatsushikaHokusai-TheGreatWaveoffKanagawa.jpeg')
path_15 = os.path.join(home, 'images/last_supper.jpg')

path_16 = os.path.join(home, 'images/fighting_temeraire.jpg')
path_17 = os.path.join(home, 'images/son_of_man.jpg')
path_18 = os.path.join(home, 'images/edward-hopper-nighthawks.jpg')
path_19 = os.path.join(home, 'images/the_new_abnormal.png')
path_20 = os.path.join(home, 'images/joan-miro-garden-lg.jpg')

path_21 = os.path.join(home, 'images/joanmiro2.jpg')

gallery_path = [path_1, path_2, path_3, path_4, path_5, path_6, path_7, path_8, path_9, path_10, path_11, path_12, path_13,
                path_14, path_15, path_16, path_17, path_18, path_19, path_20, path_21]

