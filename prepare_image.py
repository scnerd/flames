import numpy as np
import scipy.misc
import scipy.ndimage
import scipy.interpolate
from args import *

def prepare_image(condensed, **condensed_args):
    '''
    Given a linear histogram of where points have landed, render that to an image
    Condensed is a 3D matrix of shape HEIGHT x WIDTH x 4 (RGBA)
    '''
    get = make_get(condensed_args)
    gamma = get('gamma')
    vibrancy = get('vibrancy')

    alph = np.expand_dims(condensed[:,:,-1], 2)
    rgb = condensed[:,:,:-1]
    alph = alph + 1

    # Some pixels have never had a point land in them, which causes computational issues later
    # Create and store a mask of all such pixels so they can be corrected as needed
    #mask = condensed == 0
    #mask = np.prod(mask, axis=2)
    #mask = np.logical_not(np.stack([mask] * 4, axis=2))
    #if np.any(mask):
    #    condensed[mask] = condensed[mask] - np.min(condensed[mask])
    #condensed[~mask] = 0
    #mask[:,:,:-1] |= True
    #condensed[~mask] = 1

    # Scale colors by log(alpha)/alpha
    color_scale = np.log(alph) / alph
    rgb = rgb * color_scale
    alph = alph * color_scale + 1

    # Apply gamma correction
    per_color = rgb ** (1 / gamma)
    vibrancy_factors = alph ** (1 / gamma)
    vibrancy_factors = np.log(vibrancy_factors) / vibrancy_factors
    vibrant = rgb * vibrancy_factors
    rgb = vibrant * vibrancy + per_color * (1 - vibrancy)

    condensed = np.concatenate([rgb, alph], axis=2)

    # Apply saturation boost if requested
    if get('saturation') is not None or get('contrast') is not None:
        condensed = saturate(condensed, **condensed_args)

    # Resolve supersampling
    scale = 1/get('supersample')
    condensed = scipy.ndimage.zoom(condensed, zoom=(scale, scale, 1))

    return condensed

def make_renderer(verbose=False, **condensed_args):
    import theano
    import theano.tensor as T

    t_condensed = T.tensor3(dtype='float32')
    condensed = _theano_renderer(t_condensed, **condensed_args)

    renderer = theano.function([t_condensed], condensed, allow_input_downcast=True)
    def wrapper(inp, *crap, **more_crap):
        return renderer(inp)
    if verbose:
        theano.printing.pp(renderer.maker.fgraph.outputs[0])
    return wrapper

def _theano_renderer(t_condensed, **condensed_args):
    import theano
    import theano.tensor as T
    from theano.tensor.signal.pool import pool_2d

    get = make_get(condensed_args)
    gamma = get('gamma')
    vibrancy = get('vibrancy')
    downsample = get('supersample')
    downsample = (downsample, downsample)

    alph = t_condensed[:,:,-1].dimshuffle(0,1,'x')
    rgb = t_condensed[:,:,:-1]
    alph += 1.0

    color_scale = T.log(alph) / alph
    rgb = rgb * color_scale
    alph = alph * color_scale + 1

    per_color = rgb ** (1 / gamma)
    vibrancy_factors = alph ** (1 / gamma)
    vibrancy_factors = T.log(vibrancy_factors) / vibrancy_factors
    vibrant = rgb * vibrancy_factors
    rgb = vibrant * vibrancy + per_color * (1 - vibrancy)

    condensed = T.concatenate([rgb, alph], axis=2)

    # Apply saturation boost if requested
    if get('saturation') is not None or get('contrast') is not None:
        condensed = theano_saturate(condensed, **condensed_args)

    condensed = pool_2d(condensed.T, downsample, ignore_border=True).T
    return condensed
