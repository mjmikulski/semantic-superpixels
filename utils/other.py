from typing import Optional

import numpy as np


def list_stats(arr, prefix: str = Optional[None], per_feature: bool = False):
    """ Minimal list statistics """
    arr = np.asarray(arr)

    if per_feature:
        assert arr.ndim == 2, 'With per_feature flag set to True, input ' \
                              'array must have exactly 2 dimensions.'
        for i in range(arr.shape[1]):
            new_prefix = f'[{i}]' if prefix is None else f'{prefix}/{i}'
            list_stats(arr[:, i], prefix=new_prefix, per_feature=False)

        return None

    title = f"[{prefix}]\n" if prefix is not None else ''

    print_str = f"{title}" \
                f"min/median/max: " \
                f"{float(np.min(arr)):.3g}" \
                f"/{float(np.median(arr)):.3g}" \
                f"/{float(np.max(arr)):.3g} \n" \
                f"average (std): " \
                f"{(mean := float(np.mean(arr))):.3g} " \
                f"({float(np.std(arr)):.3g})"
    print(print_str)
    return mean
