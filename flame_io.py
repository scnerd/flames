import matplotlib.pyplot as plt
from args import *
from prepare_image import *
import pickle
import scipy as sc

def display_flame(condensed, renderer=prepare_image, **condensed_args):
    if condensed is None:
        raise RuntimeError("Cannot draw None")
    get = make_get(condensed_args)
    xmin = get('xmin')
    xmax = get('xmax')
    ymin = get('ymin')
    ymax = get('ymax')
    if renderer is not None:
        condensed = renderer(condensed, **condensed_args)
    fixed = condensed[:,:,:-1]
    if np.any(np.isnan(fixed)) and np.any(~np.isnan(fixed)):
        fixed[np.isnan(fixed)] = np.min(fixed[~np.isnan(fixed)])
    fixed = (fixed - fixed.min()) / fixed.ptp()
    #print(fixed)

    plt.figure(figsize=(10, 14))
    # Work-around, until I figure out what's up with pyplot
    plt.imshow(fixed, aspect='equal', interpolation='bicubic')
    #sc.misc.imsave('tmp.png', fixed, format='png')
    #plt.imshow(plt.imread('tmp.png'), aspect='equal', interpolation='nearest')
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