# Motivation
Style transfer algorithm allows sylisation of any picture with the styles of famous masterpieces. This algorithm produces especially good looking output when used on landscape pictures.

![](https://res.cloudinary.com/dbxctsqiw/image/upload/v1617573371/SmArt/213f7642f6e0111a6bd670f182305d7ffbf4305eaa7aec298762b624_mgocqr.jpg)
![](https://res.cloudinary.com/dbxctsqiw/image/upload/v1617573223/SmArt/style_content_ufu5rm.png)

Although this style transfer algorithm is impressive, it has one major limitation. Because the details of the original picture get blurred, facial features often become blurred to the point where it is difficult to recognise the person. Often, the identities of the people are the most important contents of the picture and the lack of ability of the algorithm to preserve them may prevent more people to use the algorithm.
![](https://res.cloudinary.com/dbxctsqiw/image/upload/v1617629916/SmArt/tiffany_movie_bgzfzf.png)

As seen in the above example, the style transfer worked very well. However, the girl's face certainly does not look very attractive after the facial features are blurred. 


# SmArt Style Transfer
SmArt style transfer uses PyTorch segmentation model to identify human objects and recover the details.
![](https://res.cloudinary.com/dbxctsqiw/image/upload/v1617642724/SmArt/manchester_seg_am3lfr.png)

Users can choose which object to restore.

![](https://res.cloudinary.com/dbxctsqiw/image/upload/v1617643133/SmArt/seg_outcome_manchester_brzvlk.png)

# A Step Further - Transformation GIF

Style transfer algorithms focuses on visual impact. To make the impact as strong as possible, SmArt style transfer also provides a transformation gif. It is saved in between epochs and do not add any significant computational burden.

![](https://res.cloudinary.com/dbxctsqiw/image/upload/v1617644511/SmArt/oxford-starry-full-3-1_1_1_es7lhj.gif)

# How to use
A streamlit app is developed to make SmArt simple to use.
- Clone the [repo](https://github.com/disney-snoopy/SmArtTorch).
- Run streamlit app with the following command in the repo folder.
```bash
streamlit run app_sidebar.py
```

