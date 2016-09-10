import pickle

import matplotlib.pyplot as plt
import scipy as sc

from prepare_image import *


def display_flame(condensed, renderer=prepare_image, **condensed_args):
    if condensed is None:
        raise RuntimeError("Cannot draw None")
    get = make_get(condensed_args)
    if renderer is not None:
        condensed = renderer(condensed, **condensed_args)
    fixed = condensed[:,:,:-1]
    if np.any(np.isnan(fixed)) and np.any(~np.isnan(fixed)):
        fixed[np.isnan(fixed)] = np.min(fixed[~np.isnan(fixed)])
    fixed = (fixed - fixed.min()) / fixed.ptp()

    plt.figure(figsize=(10, 14))
    plt.imshow(fixed, aspect='equal', interpolation='bicubic')
    plt.show(block=False)

def save_flame(condensed, name, transes=None, renderer=prepare_image, **condensed_args):
    data_dir = 'data/'
    if transes is not None:
        pkl = '.pkl' if not name.endswith('.pkl') else ''
        pickle.dump(transes, open(data_dir + name + pkl, 'wb'))
    if renderer is not None:
        condensed = renderer(condensed, **condensed_args)
    mask = np.isnan(condensed[:,:,:3])
    condensed[:,:,:3][mask] = np.min(condensed[:,:,:3][~mask])
    png = '.png' if not name.endswith('.png') else ''
    jpg = '.jpg' if not name.endswith('.jpg') else ''
    sc.misc.imsave(data_dir + name + png, condensed)
    sc.misc.imsave(data_dir + name + jpg, condensed)