# Skip over this, I've just put this here to get it out of the relevant code
# These functions are purely image modifications to bring out more color
# They have nothing to do with the interesting parts of rendering the fractal
from args import *
import numpy as np


def saturate(condensed, **condensed_args):
    '''
    Boost the saturation of image x by x^(1/saturation)
    '''
    saturation = make_get(condensed_args)('saturation')
    contrast = make_get(condensed_args)('contrast')

    # Convert from RGB to HSL
    size = tuple(condensed.shape[:2])
    rgb = condensed[:,:,:3]
    a = condensed[:,:,3]

    a = (a - np.min([0, np.min(a)])) / np.max([np.max(a), np.ptp(a)])
    rgb = (rgb - np.min([0, np.min(rgb)])) / np.max([np.max(rgb), np.ptp(rgb)])

    #http://www.niwa.nu/2013/05/math-behind-colorspace-conversions-rgb-hsl/
    r = rgb[:,:,0]
    g = rgb[:,:,1]
    b = rgb[:,:,2]

    mn = np.min(rgb, axis=2)
    mx = np.max(rgb, axis=2)

    l = (mn + mx) / 2

    s1 = (mx-mn)/(mx+mn)
    s2 = (mx-mn)/(2-mx-mn)
    s = np.zeros(size)
    s[l < 0.5] = s1[l < 0.5]
    s[l >= 0.5] = s2[l >= 0.5]

    h = np.zeros(size)
    r_max = r == mx
    g_max = g == mx
    b_max = b == mx
    h[r_max] = 0 + (g[r_max] - b[r_max]) / (mx[r_max] - mn[r_max])
    h[g_max] = 2 + (b[g_max] - r[g_max]) / (mx[g_max] - mn[g_max])
    h[b_max] = 4 + (r[b_max] - g[b_max]) / (mx[b_max] - mn[b_max])
    h /= 6
    h[np.isnan(h)] = 0

    # Boost channels
    if saturation is not None:
        s = s ** (1 / saturation)
    if contrast:
        half = np.ptp(l) / 2
        mean = np.min(l) + half
        l = (l - mean) / half
        l[l > 0] = l[l > 0] ** (1 / contrast)
        l[l < 0] = -((-l[l < 0]) ** (1 / contrast))
        l = l * half + mean

    # Convert from HSL to RGB
    #http://www.niwa.nu/2013/05/math-behind-colorspace-conversions-rgb-hsl/
    t11 = l * (1 + s)
    t12 = l + s - l * s
    t1 = np.zeros(size)
    t1[l < 0.5] = t11[l < 0.5]
    t1[l >= 0.5] = t12[l >= 0.5]
    t2 = 2 * l - t1

    def lim(t):
        t = np.copy(t)
        t[t > 1] -= 1
        t[t < 0] += 1
        return t
    tr = lim(h + (1/3))
    tg = lim(h)
    tb = lim(h - (1/3))

    def run_test(tmp):
        test1 = (6 * tmp) <= 1
        test2 = ~test1 & ((2 * tmp) <= 1)
        test3 = ~test1 & ~test2 & (3 * tmp <= 2)
        res = np.ones(size) * t2
        res[test1] = t2[test1] + (t1[test1] - t2[test1]) * 6 * tmp[test1]
        res[test2] = t1[test2]
        res[test3] = t2[test3] + (t1[test3] - t2[test3]) * (2/3 - tmp[test3]) * 6
        res[s <= 0] = l[s <= 0]
        return res
    r = run_test(tr)
    g = run_test(tg)
    b = run_test(tb)

    return np.stack([r, g, b, a], axis=2)

def theano_saturate(t_condensed, **condensed_args):
    '''
    Boost the saturation of image x by x^(1/saturation)
    '''
    import theano.tensor as T
    saturation = make_get(condensed_args)('saturation')
    contrast = make_get(condensed_args)('contrast')

    # Convert from RGB to HSL
    rgb = t_condensed[:,:,:-1]
    a = t_condensed[:,:,-1]

    a = (a - T.min([0, T.min(a)])) / T.max([T.max(a), T.ptp(a)])
    rgb = (rgb - T.min([0, T.min(rgb)])) / T.max([T.max(rgb), T.ptp(rgb)])

    #http://www.niwa.nu/2013/05/math-behind-colorspace-conversions-rgb-hsl/
    #mn = T.min([0, T.min(rgb)])
    #ptp = T.max([T.max(rgb), T.ptp(rgb)])
    #rgb = (rgb-mn) / ptp
    r = rgb[:,:,0]
    g = rgb[:,:,1]
    b = rgb[:,:,2]

    mn = T.min(rgb, axis=2)
    mx = T.max(rgb, axis=2)

    l = (mn + mx) / 2

    s1 = (mx-mn)/(mx+mn)
    s2 = (mx-mn)/(2-mx-mn)
    s = T.switch(T.lt(l, 0.5), s1, s2)

    h = T.zeros_like(a)
    r_max = T.eq(r, mx)
    g_max = T.eq(g, mx)
    b_max = T.eq(b, mx)
    h1 = 0 + (g - b) / (mx - mn)
    h2 = 2 + (b - r) / (mx - mn)
    h3 = 4 + (r - g) / (mx - mn)
    h = T.switch(r_max, h1, T.switch(g_max, h2, T.switch(b_max, h3, h)))
    h /= 6
    h = T.switch(T.isnan(h), 0, h)

    # Boost channels
    if saturation is not None:
        s = s ** (1 / saturation)
    if contrast:
        half = T.ptp(l) / 2
        mean = T.min(l) + half
        l = (l - mean) / half
        l1 = l ** (1 / contrast)
        l2 = -((-l) ** (1 / contrast))
        l = T.switch(T.lt(l, 0), l2, l1)
        l = l * half + mean

    # Convert from HSL to RGB
    #http://www.niwa.nu/2013/05/math-behind-colorspace-conversions-rgb-hsl/
    t11 = l * (1 + s)
    t12 = l + s - l * s
    t1 = T.switch(T.lt(l, 0.5), t11, t12)
    t2 = 2 * l - t1

    def lim(t):
        t = t.copy()
        t = T.switch(T.gt(t, 1), t-1, t)
        t = T.switch(T.lt(t, 1), t+1, t)
        return t
    tr = lim(h + (1/3))
    tg = lim(h)
    tb = lim(h - (1/3))

    def run_test(tmp):
        test1 = T.le((6 * tmp), 1)
        test2 = ~test1 & T.le((2 * tmp), 1)
        test3 = ~test1 & ~test2 & T.le((3 * tmp), 2)
        res = T.ones_like(tmp) * t2
        res1 = t2 + (t1 - t2) * 6 * tmp
        res2 = t1
        res3 = t2 + (t1 - t2) * (2/3 - tmp) * 6
        res = T.switch(test1, res1, T.switch(test2, res2, T.switch(test3, res3, res)))
        res = T.switch(T.le(s, 0), l, res)
        return res
    r = run_test(tr)
    g = run_test(tg)
    b = run_test(tb)

    return T.stack([r, g, b, a], axis=2)