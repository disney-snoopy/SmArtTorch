# SmArt Style Transfer
- Torch implementation of style transfer
- Segmentation model is used in series to detect and restore human objects from the stylised output
- Docker / streamlit deployment for public access
- 
![Alt text](https://res.cloudinary.com/dbxctsqiw/image/upload/v1617573371/SmArt/213f7642f6e0111a6bd670f182305d7ffbf4305eaa7aec298762b624_mgocqr.jpg "a title")
![Alt text](https://res.cloudinary.com/dbxctsqiw/image/upload/v1617573223/SmArt/style_content_ufu5rm.png "a title")


## Motivation
Style transfer allows amazing stylisation of any pictures with styles from masterpieces. Unfortunately, style transfer tends to blur the details of the original picture that a user might want to preserve. The limitation becomes very visible when the details of interest are not dominantly picked up by convolution filters. I stylsed my wife's picture to give her a pleasant surprise, only to be disappointed by her blurry face in the final output. To solve this problem, I implemented segmentation model to detect human objects in the picture and restore the details of them after stylisation.



## How to use
After cloning the repo, you can run the streamlit app either with your local backend or remote backend.

```bash
streamlit run app_sidebar.py
```


