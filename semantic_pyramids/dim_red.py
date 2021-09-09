import numpy as np
from sklearn.decomposition import PCA

from utils.other import list_stats


def resize_and_flatten(feature_map: np.ndarray):
    feature_map = np.moveaxis(feature_map, 0, -1)
    feature_map = np.reshape(feature_map, (-1, feature_map.shape[-1]))
    return feature_map


def dim_red(features: np.ndarray, num_components: int = 3, as_uint8: bool = False):
    pca = PCA(num_components, whiten=True)
    features = pca.fit_transform(features)
    print('Explained variance: ', pca.explained_variance_ratio_)

    # robust scaling for visualization
    q05 = np.percentile(features, 5, axis=0)
    q95 = np.percentile(features, 95, axis=0)
    scale = q95 - q05
    m = np.median(features, axis=0)

    z2 = (features - m) / scale
    list_stats(z2, 'scaled', per_feature=True)

    if as_uint8:
        return np.clip((z2 + 0.5) * 255, 0, 255).astype('uint8')

    return z2
