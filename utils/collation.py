import warnings
from typing import Dict, Iterable, Union

import cv2
import numpy as np

DEFAULT_MASK_COLOR = (255, 255, 255)
CAPTION_COLOR = (0, 150, 250)
CAPTION_COLOR_ALT = (85, 95, 105)


def _get_consistent_shape(images: Iterable):
    """ Check if images have the same shape and if so, return it. """
    dim0s = []
    dim1s = []

    for img in images:
        dim0s.append(img.shape[0])
        dim1s.append(img.shape[1])

    assert len(set(dim0s)) == 1 and len(set(dim1s)) == 1, 'Inconsistent shapes.'

    return dim0s[0], dim1s[0]


def _find_grid(n, dim0, dim1, ratio=1.7, verbose=False):
    """ Find a grid (k0 x k1) that:
    * can fit in `n` images, i.e. `k0` * `k1` >= `n`,
    * has overall ratio close to `ratio`,
    * has minimal free space.

    Parameters
    ----------
    n: int
        Number of images.
    dim0, dim1
        Height and width of each image.
    ratio: float, optional
        Desirable ratio of final grid.
    verbose: bool, optional
        If True, prints shape and baddness of the found grid.

    Returns
    -------
    k0, k1 - number rows and columns

    Examples
    --------
    >>> _find_grid(20, 1, 8, ratio=1.73, verbose=True)
    badness -0.98 for grid 10 x 2 for 20 images with ratio 1.6 (intended: 1.73)
    (10, 2)
    """
    k0 = k1 = 1

    def badness(k0, k1):
        ratio_badness = ((k1 * dim1) / (k0 * dim0) - ratio) ** 2
        free_space_badness = (k0 * k1 / n) ** 2 - 1 if k0 * k1 > n else 0
        exact_bonus = -1 if k0 * k1 == n else 0
        return ratio_badness + free_space_badness + exact_bonus

    while k0 * k1 < n:
        b0 = badness(k0 + 1, k1)
        b1 = badness(k0, k1 + 1)
        if b0 < b1:
            k0 += 1
        else:
            k1 += 1

    if verbose:
        print(f'badness {badness(k0, k1):.2g} for grid {k0} x {k1} for {n} images '
              f'with ratio {(k1 * dim1) / (k0 * dim0):.3g} (intended: {ratio})')

    return k0, k1


def homogenize_image(img: np.ndarray, mask_color=DEFAULT_MASK_COLOR):
    """ Homogenization steps:

    * Filter out all possible shape problems.
    * Represent grayscale image as color image (i.e. with 3 channels).
    * Transfer float images to uint8.
    * Recolor binary images, e.g. masks.
    * Warn if the image has low contrast.

    Parameters
    ----------
    img: np.ndarray
        Image to homogenize.
    mask_color: tuple or np.ndarray, optional
        Color to use in case of masks.

    Returns
    -------
    Image where img.shape[2] == 3 and dtype == uint8.
    """
    img = np.squeeze(img)
    mask_color = np.asarray(mask_color, dtype='uint8')

    rank = len(img.shape)
    if rank == 3:  # color image
        if img.shape[2] == 4:
            raise ValueError('No handling of images with alpha channel')
        if img.shape[2] == 2:
            raise ValueError('No handling of grey images with alpha channel')
    elif rank == 2:  # black and white image
        img = np.stack((img, img, img), axis=2)
    else:
        raise ValueError('Wrong shape of the image')

    if img.dtype != np.dtype('uint8'):
        if np.min(img) < 0 or np.max(img) > 1:
            raise ValueError('Image values not uin8 and not within [0,1].')
        img = (img * 255).astype('uint8')

    if (np.max(img) - np.min(img)) <= 5:
        if len(unique_values := np.unique(img)) == 2:
            img[img == unique_values[0]] = 0
            img[img == unique_values[1]] = 1
            img = img * mask_color
        else:
            warnings.warn('Low contrast image')

    assert img.shape[2] == 3
    assert img.dtype == 'uint8'

    return img


def add_caption(img: np.ndarray, caption: str) -> np.ndarray:
    """Add a caption to an image.

    The text will be located in the bottom-left corner of the image.

    Parameters
    ----------
    img : np.ndarray
        Image to which to add the caption.
    caption: str
        Text to place on the image. Whenever the symbol "#" is included in the text an alternative
        color will be used rather than the default one.

    Returns
    -------
        Image with the added caption.
    """
    img = img.copy()

    eps = (img.shape[0] + img.shape[1]) // 100
    text_location = (eps, img.shape[0] - eps)

    if '#' in caption:
        text_color = CAPTION_COLOR_ALT
    else:
        text_color = CAPTION_COLOR

    cv2.putText(img, caption,
                org=text_location,
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=2,
                color=text_color,
                thickness=2,
                lineType=cv2.LINE_AA)
    return img


def collate_images(images: Union[Dict[str, np.ndarray], Iterable[np.ndarray]],
                   skip_commented: bool = False) -> np.ndarray:
    """ Combine multiple images into one and add captions.

    It tries to find a grid of images that looks good on a screen,
    i.d. has minimal wasted space.

    Parameters
    ----------
    images: dict or iterable
        Dictionary where keys are captions that will be written on
        images and values are images in form of numpy arrays of the same
        shape. If not a dict, will be converted to a dict with names
        being consecutive numbers.
    skip_commented: bool
        If True, all entries in the images dict that contain '#' in the
        key, will be dropped.

    Returns
    -------
    Collated images as numpy array, with uint8 dtype.
    """
    if not isinstance(images, dict):
        images = {f'{i:03d}': img for i, img in enumerate(images)}

    if skip_commented:
        images = {k: v for k, v in images.items() if '#' not in k}

    n = len(images)
    dim0, dim1 = _get_consistent_shape(images.values())
    k0, k1 = _find_grid(n, dim0, dim1)

    output = np.zeros((k0 * dim0, k1 * dim1, 3), dtype='uint8')

    for i, (name, img) in enumerate(images.items()):
        pos0 = i // k1
        pos1 = i % k1
        img = homogenize_image(img)
        img = add_caption(img, caption=name)
        output[pos0 * dim0:(pos0 + 1) * dim0, pos1 * dim1:(pos1 + 1) * dim1, :] = img

    return output
