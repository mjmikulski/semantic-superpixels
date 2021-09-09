import os

import cv2 as cv
import numpy as np
import torch as T
from torchvision.transforms import functional as F

from settings import IMAGES_DIR


def load_img(filename: str,
             resize: float = None,
             images_dir=IMAGES_DIR) -> np.ndarray:
    file = os.path.join(images_dir, filename)
    img = cv.imread(file, cv.IMREAD_COLOR + cv.IMREAD_IGNORE_ORIENTATION)

    if resize is not None:
        img = cv.resize(img, (0, 0), fx=resize, fy=resize, interpolation=cv.INTER_AREA)

    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = img.astype('uint8')

    return img


def img_to_torch_input(img: np.ndarray) -> T.Tensor:
    img = np.asarray(img)
    img = np.moveaxis(img, -1, 0)
    img = img.astype('float32') / 255
    img = img[np.newaxis, ...]
    img = T.from_numpy(img)
    img = F.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    return img
