from flame_io import *
from condense import *
from prepare_image import *
from prepare_image import _theano_renderer
import theano.sandbox.cuda as cuda
from tqdm import tqdm

theanoify = True
use_gpu = True

def chaos_game_cpu(fs, f_final=None, n=1000000, n_runs=1, iters=200, min_iter=20, and_display=True, resume=None, seed=None, **condense_args):
    '''
    The Chaos Game is simple (https://en.wikipedia.org/wiki/Chaos_game)
    1) Pick a random point (x,y)
    2) Apply a random transform F to it
    3) Record where it lands (in terms of pixels)
    4) Repeat a lot
    The Flame algorithm tweaks this to include 3 color channels per point
    This results in four histograms, one for each color channel, one for alpha (alpha for a point is 1)
    This implementation is a vectorized approach that runs n points simultaneously
    Note that the result is the same, since for a single iteration, the function applied to a point
    is still randomly chosen per-point
    '''
    condense_args['n'] = n

    condensed = resume
    fs = [(var, weight) for f in fs for var, weight in f]
    fs, weights = zip(*fs)
    if theanoify:
        import theano
        import theano.tensor as T
        t_x = T.vector(dtype='float32')
        t_y = T.vector(dtype='float32')
        fs = [f(t_x, t_y) for f in fs]
        fs = [theano.function([t_x, t_y], [out_x, out_y], allow_input_downcast=True) for out_x, out_y in fs]
    # Launching with the same parameters should result in the same color scheming
    np.random.seed((hash(tuple(fs)) % 0xffffffff) if seed is None else seed)
    weights /= np.sum(weights)
    num_fs = len(fs)
    range_fs = list(range(num_fs))
    # Assign a random color to each transform
    colors = np.random.uniform(low=0, high=1, size=(num_fs, 3))
    colors = np.hstack([colors, np.ones((num_fs, 1))])
    color_final = np.random.uniform(low=0, high=1, size=3)
    color_final = np.hstack([color_final, [1]])

    np.random.seed()
    condense = make_condense(**condense_args)
    from time import time
    for run in range(n_runs):
        # Pick n random points
        x = np.random.uniform(low=-1.0, high=1.0, size=n)
        y = np.random.uniform(low=-1.0, high=1.0, size=n)
        xf = np.copy(x)
        yf = np.copy(y)
        # Assign each a random starting color
        c = np.random.uniform(low=0,    high=1.0, size=(n, 4))
        cf = np.copy(c)
        for it in tqdm(range(iters)):
            s1 = time()
            # Pick a transform for each point
            f_choices = np.random.choice(range_fs, p=weights, replace=True, size=n)
            for f in range(num_fs):
                # For all points affected by transform f, apply f
                sel = f_choices == f
                x[sel], y[sel] = fs[f](x[sel], y[sel])
                if f_final is not None:
                    xf[sel], yf[sel] = f_final(x[sel], y[sel])
                else:
                    xf[sel], yf[sel] = x[sel], y[sel]
                # Shift these points' colors toward f's color
                c[sel] = (c[sel] + colors[f]) / 2
                # Shift skew toward a universal color
                cf[sel] = (c[sel] + color_final) / 2
            s2 = time()
            # Render this iteration of points
            if it >= min_iter:
                condensed = condense(xf, yf, cf) + (condensed if condensed is not None else 0)
            s3 = time()
            print("Times: %f, %f" % (s2-s1, s3-s2))
            if type(and_display) == int and it >= min_iter and it % and_display == 0:
                if use_gpu:
                    condensed = buffer.sum(0).get()
                display_flame(condensed, **condense_args)
    if and_display:
        display_flame(condensed, **condense_args)
    return condensed

def chaos_game(fs, f_final=None, n=1000000, n_runs=1, iters=200, min_iter=20, and_display=True, resume=None, reuse_func=None, seed=None, **condense_args):
    '''
    The Chaos Game is simple (https://en.wikipedia.org/wiki/Chaos_game)
    1) Pick a random point (x,y)
    2) Apply a random transform F to it
    3) Record where it lands (in terms of pixels)
    4) Repeat a lot
    The Flame algorithm tweaks this to include 3 color channels per point
    This results in four histograms, one for each color channel, one for alpha (alpha for a point is 1)
    This implementation is a vectorized approach that runs n points simultaneously
    Note that the result is the same, since for a single iteration, the function applied to a point
    is still randomly chosen per-point
    '''
    import theano
    import theano.tensor as T
    #color_random = lambda size: np.random.uniform(low=0, high=1, size=size)
    color_random = lambda size: np.random.beta(0.3, 0.3, size=size)

    cast = np.cast['float32']

    fs = [(var, weight) for f in fs for var, weight in f]
    num_fs = len(fs)
    fs, weights = zip(*fs)
    print(weights)
    weights /= np.sum(weights)

    if reuse_func is None:
        condense_args['n'] = n
        if resume is not None:
            condensed = theano.shared(resume)
        else:
            condensed = theano.shared(cuda.CudaNdarray(cast(np.zeros((1,1,1,1)))), name='condensed')

        np.random.seed((hash(tuple(fs)) % 0xffffffff) if seed is None else seed)
        # Assign a random color to each transform
        colors = color_random((num_fs, 3))
        colors = np.hstack([colors, np.ones((num_fs, 1))])
        colors = cuda.CudaNdarray(cast(colors), dtype='float32')

        color_final = color_random(3)
        color_final = np.hstack([color_final, [1]])
        color_final = cuda.CudaNdarray(cast(color_final), dtype='float32')

        np.random.seed(seed)
        # Pick n random points
        x = np.random.uniform(low=-1.0, high=1.0, size=n)
        y = np.random.uniform(low=-1.0, high=1.0, size=n)
        # Assign each a random starting color
        #c = np.random.uniform(low=0,    high=1.0, size=(n, 4))
        c = color_random((n,4))

        t_x = theano.shared(cuda.CudaNdarray(cast(x), dtype='float32'), name='x')
        t_y = theano.shared(cuda.CudaNdarray(cast(y), dtype='float32'), name='y')
        t_c = theano.shared(cuda.CudaNdarray(cast(c), dtype='float32'), name='c')

        for f in fs:
            __x, __y = f(t_x, t_y)
            print(str(f))
            print("{}, {}".format(__x.type(), __y.type()))

        condensor_op = PyCUDAHist2D(**condense_args)

        t_choices = T.ivector('choices')

        #new_x, new_y, new_c = t_x, t_y, t_c
        new_xs, new_ys, new_cs = [], [], []
        for f in range(num_fs):
            sel = T.eq(t_choices, f).nonzero()
            tmp_x, tmp_y = fs[f](t_x[sel], t_y[sel])
            new_xs.append(tmp_x)
            new_ys.append(tmp_y)
            new_cs.append((t_c[sel] + colors[f]) / 2)

        new_x = T.concatenate(new_xs, axis=0)
        new_y = T.concatenate(new_ys, axis=0)
        new_c = T.concatenate(new_cs, axis=0)

        cf = (new_c + color_final) / 2
        t_condense = condensor_op(new_x, new_y, cf)
        t_condense = theano.ifelse.ifelse(condensed.size > 1, t_condense + condensed, t_condense)
        t_condense_final = t_condense.sum(axis=0, keepdims=False)
        rendered = _theano_renderer(t_condense_final, **condense_args)

        pre_step_function = theano.function([t_choices], [], updates=[(t_x, new_x), (t_y, new_y), (t_c, new_c)])
        step_function = theano.function([t_choices], [], updates=[(condensed, t_condense), (t_x, new_x), (t_y, new_y), (t_c, new_c)])
        _create_image = theano.function([t_condense], [rendered])
        reuse_func = (pre_step_function, step_function, _create_image, condensed, t_x, t_y, t_c)
    else:
        pre_step_function, step_function, _create_image, condensed, t_x, t_y, t_c = reuse_func
        condensed.set_value(cuda.CudaNdarray(cast(np.zeros((1,1,1,1)))))
        t_x.set_value(cuda.CudaNdarray(cast(np.random.uniform(low=-1.0, high=1.0, size=n)), dtype='float32'))
        t_y.set_value(cuda.CudaNdarray(cast(np.random.uniform(low=-1.0, high=1.0, size=n)), dtype='float32'))
        t_c.set_value(cuda.CudaNdarray(cast(color_random((n,4))), dtype='float32'))

    create_image = lambda: np.asarray(_create_image(condensed.get_value()))[0,:,:,:]

    #theano.printing.debugprint(step_function)

    range_fs = range(num_fs)
    for run in range(n_runs):
        for it in tqdm(range(iters)):
            f_choices = np.random.choice(range_fs, p=weights, replace=True, size=n)
            if it < min_iter:
                pre_step_function(f_choices)
            else:
                step_function(f_choices)
            if type(and_display) == int and it >= min_iter and it % and_display == 0:
                res = create_image()
                display_flame(res, renderer=None, **condense_args)
    if (type(and_display) == bool and and_display) or (type(and_display) == int and it >= min_iter and it % and_display == 0):
        display_flame(create_image(), renderer=None, **condense_args)
    return condensed.get_value(), create_image(), reuse_func

#test_settings = {
#    'n': 1000000,
#    'n_runs': 1,
#    'iters': 50,
#    'xmin': -1.166,
#    'xmax': 1.166,
#    'ymin': -1,
#    'ymax': 1,
#    'and_display': True,
#    'xres': 1440,
#    'yres': 1080,
#    'supersample': 2,
#    'vibrancy': 1,
#    'saturation': 1,
#}
#transes = [rotate(np.pi)]
#cond, img = chaos_game(transes, **test_settings)