import cv2
import numpy as np
import torch as T
import torchvision as TV
from sklearn.decomposition import PCA

from semantic_pyramids.dim_red import resize_and_flatten
from utils.io_utils import img_to_torch_input


class FeatureExtractor:
    def __init__(self, model: T.nn.Module):
        self.model = model
        self.feature_maps = dict()
        self.register_hooks_for_all_layers(model)

    def get_hook(self, name: str):
        def hook(model, input, output):
            self.feature_maps[name] = np.squeeze(output.detach().numpy())

        return hook

    def register_hooks_for_all_layers(self, model: T.nn.Module, prefix=None):
        for name, layer in model._modules.items():
            if isinstance(layer, T.nn.Sequential):
                self.register_hooks_for_all_layers(layer, prefix=name)
            else:
                layer.register_forward_hook(self.get_hook(f'{prefix}:{name}' if prefix else name))

    def forward(self, x: T.Tensor) -> dict[str, np.ndarray]:
        self.model(x)
        return self.feature_maps


def get_features(orig_img_rgb: np.ndarray, layers_with_num: dict, use_orig_img: bool = False):
    model = TV.models.resnet152(pretrained=True)
    model.eval()
    feature_extractor = FeatureExtractor(model)
    orig_shape = orig_img_rgb.shape[:2]
    img = img_to_torch_input(orig_img_rgb)
    feature_maps = feature_extractor.forward(img)

    maps = []

    if use_orig_img:
        fmap = cv2.cvtColor(orig_img_rgb, cv2.COLOR_RGB2LAB) / 255 - 0.5
        maps.append(fmap)

    for name, fmap in feature_maps.items():
        if name not in layers_with_num.keys() or fmap.ndim != 3:
            continue

        features = resize_and_flatten(fmap)
        num_dim = layers_with_num[name]
        pca = PCA(num_dim, whiten=True)
        features = pca.fit_transform(features)
        print('Explained variance: ', pca.explained_variance_ratio_)

        fmap_reduced = np.reshape(features, (*fmap.shape[1:], num_dim))
        fmap_reduced = cv2.resize(fmap_reduced, orig_shape[::-1], interpolation=cv2.INTER_LINEAR)
        if fmap_reduced.ndim < 3:
            fmap_reduced = fmap_reduced[..., np.newaxis]

        maps.append(fmap_reduced)

    final_map = np.concatenate(maps, axis=-1)

    return final_map
