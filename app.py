import numpy as np
import streamlit as st
from skimage.segmentation import mark_boundaries, slic

from constants import RESNET152_LAYERS
from semantic_pyramids.features import get_features
from utils.st_utils import EXPLANATION, img_picker, load_img

DEFAULT_LAYERS = ['layer2:7', 'layer3:21', 'layer3:35']

st.set_page_config(layout='wide',
                   initial_sidebar_state='expanded',
                   page_title='Semantic Superpixels')

st.title('Hyper pixels')

left, right = st.columns(2)

right.markdown(EXPLANATION)

st.sidebar.markdown("""## Input image""")
filename = img_picker()
img_scale = st.sidebar.number_input('scale', 0.1, 1.0, 0.3, 0.1,
                                    help='How much to down scale the image '
                                         'before processing.Choose smaller '
                                         'values for faster processing.')
orig_img = load_img(filename, img_scale)
left.image(orig_img, caption='Input image')

st.sidebar.markdown("""\
--- 
## Features
Pick ResNet layers' outputs to use. 
Then define how many principal components of each should be taken.
""")
layers = st.sidebar.multiselect('Feature layers used for clustering',
                                RESNET152_LAYERS, default=DEFAULT_LAYERS)
use_orig = st.sidebar.checkbox('Use orig image as well',
                               help='If checked, the orig image in LAB '
                                    'colorspace will be concatenated to '
                                    '(scaled up) ResNet features.')
layers_with_num = {}

for layer in layers:
    num = st.sidebar.number_input(f'num components of {layer}', 1, 32, 6, 1, )
    layers_with_num[layer] = int(num)

st.sidebar.markdown("""\
--- 
## SLIC
Here you can tweak some SLIC parameters.
""")

num_clusters_ticks = [50, 200, 1000, 4000]
num_clusters = st.sidebar.select_slider('num_clusters',
                                        options=num_clusters_ticks, value=200)

dist_weight_ticks = [0.1, 0.3, 1, 3, 10, 30]
compactness_ssp = st.sidebar.select_slider('semantic superpixels compactness',
                                           options=dist_weight_ticks, value=1)
compactness_sp = st.sidebar.select_slider('superpixels compactness',
                                          options=dist_weight_ticks, value=10)

fmaps = get_features(orig_img, layers_with_num, use_orig)

segments = slic(fmaps,
                n_segments=num_clusters,
                compactness=compactness_ssp,
                multichannel=True,
                convert2lab=False,
                start_label=1,
                min_size_factor=0.1,
                max_size_factor=3
                )
slic_img = mark_boundaries(orig_img, segments, outline_color=(0, 0, 0.5))
left.image(slic_img, caption=f'Semantic superpixels [{np.max(segments)}]')

segments = slic(orig_img,
                n_segments=num_clusters,
                compactness=compactness_sp,
                multichannel=True,
                convert2lab=True,
                start_label=1,
                min_size_factor=0.1,
                max_size_factor=3
                )
slic_img = mark_boundaries(orig_img, segments, outline_color=(0, 0, 0.5))
left.image(slic_img, caption=f'Regular superpixels [{np.max(segments)}]')
