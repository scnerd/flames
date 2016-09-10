from chaos_game import *

class Var():
    '''
    Every variation takes the form of the following function:
    F(x, y) = V_p(a*x + b*y + c, d*x + e*y + f)
    Where V_p is some parametrized function that takes 2D points and returns 2D points
    Note that V_p does not have to be linear (and most of the ones below aren't)
    This class simplifies the definition of new variations by implementing the a-f transformation
    step, then calling V_p, while also storing what a-f and the parameters p are, thus allowing
    this class to be called simply as "my_custom_variation(x, y)", and all other parameters are
    automatically made available to V_p
    Also, since Python allows "func(*params)", you can name your extra parameters whatever you want
    E.g., def my_custom_var(x, y, a, b, c, d, e, f, theta, epsilon, foobar): ...
    Use the @variation decorator below to create Var instances from V_p functions
    NOTE: When implementing V_p, DO NOT perform the a*x+b*y+...+f transform, this is pre-applied
    to x, y before they get passed to V_p
    '''
    def __init__(self, func, num_params, a, b, c, d, e, f, *p, **kwargs):
        dargs = np.random.randn(6)
        self.args = [a, b, c, d, e, f]
        self.args = [v if v is not None else dv for v, dv in zip(self.args, dargs)]
        p = [p[i] if i < len(p) else np.random.randn() for i in range(num_params)]
        self.p = list(p)
        self.all_params = self.args + self.p
        if theanoify:
            import theano.tensor as T
            self.shared_vars = [theano.shared(np.cast['float32'](p), l, allow_downcast=True) for l, p in zip('abcdef', self.all_params)]
            s = self.shared_vars
            def shared_pre(x, y):
                return s[0] * x + s[1] * y + s[2], s[3] * x + s[4] * y + s[5]
            self.shared_pre = shared_pre
        self.func = func #jit(func)
        self.make_pre()
        self.weight = 1
        self.unchanging = kwargs.get('unchanging', False)
        #print("Initialized %s with %d parameters (%s)" % (func.__name__, num_params, ",".join("%f" % param for param in self.all_params)))

    def make_pre(self):
        if theanoify:
            self.all_params = np.cast['float32'](self.all_params)
            for shared_var, p_value in zip(self.shared_vars, self.all_params):
                shared_var.set_value(p_value)
        else:
            self.all_params = np.array(self.all_params)
            a, b, c, d, e, f = self.all_params[:6]
            def pre(x, y):
                return a * x + b * y + c, d * x + e * y + f
            self.pre = pre

    def __call__(self, x, y):
        if theanoify:
            x, y = self.shared_pre(x, y)
            return self.func(x, y, *self.shared_vars)
        else:
            x, y = self.pre(x, y)
            return self.func(x, y, *self.all_params)

    def __str__(self):
        return "%s (%s)" % (self.func.__name__,
                            ",".join(str(param) for param in self.all_params))

    def __len__(self):
        return 1

    def weights(self):
        return [float(self.weight)]

    def __iter__(self):
        return iter([(self, float(self.weight))])

    def __hash__(self):
        return hash(tuple([self.func.__name__] + list(self.all_params)))

    def visualize(self, default_params=True):
        num = 30
        xs = np.linspace(-1, 1, num=num, endpoint=True)
        ys = xs
        xs, ys = np.meshgrid(xs, ys)
        if not default_params:
            xs, ys = self.pre(xs, ys)
        all_args = ([1,0,0,0,1,0] + [1 for p in self.p]) if default_params else self.all_args
        xs, ys = self.func(xs, ys, *all_args)
        ys *= -1 # To match how images are displayed
        fig = plt.figure(figsize=(4,4))
        style = 'b-'
        for col in range(num):
            plt.plot(xs[:,col], ys[:,col], style)
        style = 'r-'
        for row in range(num):
            plt.plot(xs[row,:], ys[row,:], style)
        plt.title(self.func.__name__)
        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)
        plt.show(block=False)

    def sample(self, **kwargs):
        '''
        This doesn't really work well... need to find a way to demonstrate a variation
        without all the points exploding away from [-1,1], and with something reliably good
        '''
        default_args = {'xres': 640, 'yres': 480, 'supersample': 1, 'and_display': True, 'n': 10000, 'iters': 100}
        default_args.update(**kwargs)
        print(self.func.__name__)
        # Repeating itself allows for multiple colors to arise
        chaos_game([self]*3, **default_args)

    def __getstate__(self):
        return {'p': self.p, 'a': self.args, 'f': self.func.__name__}

    def __setstate__(self, state):
        self.p = state['p']
        self.args = state['a']
        self.all_params = self.args + self.p
        a, b, c, d, e, f = self.args
        self.func = _all_variations[state['f']]
        def pre(x, y):
            return a * x + b * y + c, d * x + e * y + f
        self.pre = pre


_all_variations = {}
all_variations = []
def variation(func_or_num_params=None):
    '''
    This is a function decorator that can optionally be parametrized
    @variation == @variation(0), so I'll just describe @variation(n)
    Use variation to decorate a function with the following signature:
    def my_custom_var(x, y, a, b, c, d, e, f, ...)
    variation converts this into the equivalent of the following signature:
    def initialize_my_custom_var(a, b, c, d, e, f, ...):
        def my_custom_var(x, y):
            ...
        return my_custom_var
    This allows you to initialize parametric variations, and then use them as simple 2D transforms
    '''
    if isinstance(func_or_num_params, int):
        num_params = func_or_num_params
        func = None
    else:
        num_params = 0
        func = func_or_num_params

    if func is None:
        def make_variation(func):
            def inner(a=None, b=None, c=None, d=None, e=None, f=None, *p, **kwargs):
                return Var(func, num_params, a, b, c, d, e, f, *p, **kwargs)
            _all_variations[func.__name__] = func
            all_variations.append(inner)
            return inner
        return make_variation
    else:
        def inner(a=None, b=None, c=None, d=None, e=None, f=None, *p, **kwargs):
            return Var(func, num_params, a, b, c, d, e, f, *p, **kwargs)
        _all_variations[func.__name__] = func
        all_variations.append(inner)
        return inner

# ==================================================
# The actual transforms
# ==================================================
if theanoify:
    import theano
    import theano.tensor as T
    from theano.tensor.shared_randomstreams import RandomStreams
    _rand_stream = RandomStreams()
    sin = T.sin
    cos = T.cos
    pi = np.cast['float32'](np.pi)
    arctan = T.arctan2
    sqrt = T.sqrt
    choice = _rand_stream.choice
    uniform = _rand_stream.uniform
    def _trunc(v):
        raise NotImplementedError()
    def _zeros(x):
        return T.zeros_like(x, dtype='float32')
else:
    sin = np.sin
    cos = np.cos
    pi = np.pi
    arctan = np.arctan2
    sqrt = np.sqrt
    choice = np.random.choice
    uniform = np.random.uniform
    def _trunc(v):
        return np.trunc(v)
    def _zeros(x):
        return np.zeros(_size(x))
def _r2(x, y):
    return x * x + y * y
def _r(x, y):
    return sqrt(_r2(x, y))
def _theta(x, y):
    return arctan(x, y)
def _phi(x, y):
    return arctan(y, x)
def _size(x):
    return x.shape[0]
def _omega(x):
    return choice([0, pi], size=_size(x))
def _alpha(x):
    return choice([-1, 1], size=_size(x))
def _unif(x):
    return uniform(low=0, high=1, size=_size(x))

@variation # can also be @variation() or @variation(0)
def identity(x, y, a, b, c, d, e, f):
    return x, y

rotate = lambda theta: identity(np.cos(theta), -np.sin(theta), 0, np.sin(theta), np.cos(theta), 0, unchanging=True)
dihedral = lambda: identity(-1, 0, 0, 0, 1, 0, unchanging=True)

#@variation(1)
#def _free_rotation(x, y, a, b, c, d, e, f, *theta):
#    theta = theta[0]
#    return cos(theta) * x - sin(theta) * y, sin(theta) * x + cos(theta) * y

@variation
def sinusoidal(x, y, a, b, c, d, e, f):
    return sin(x), cos(y)

@variation
def spherical(x, y, a, b, c, d, e, f):
    ir = 1 / _r2(x, y)
    return ir * x, ir * y

@variation
def pdj(x, y, a, b, c, d, e, f):
    return sin(a * y) - cos(b * x), sin(c * x) - cos(d * y)

@variation
def handkerchief(x, y, a, b, c, d, e, f):
    r = _r(x, y)
    t = _theta(x, y)
    return r * sin(t + r), r * cos(t - r)

@variation
def disc(x, y, a, b, c, d, e, f):
    td = _theta(x, y) / pi
    r = _r(x, y)
    return td * sin(pi * r), td * cos(pi * r)

@variation
def polar(x, y, a, b, c, d, e, f):
    r = _r(x, y)
    t = _theta(x, y)
    return t / pi, r - 1

@variation
def swirl(x, y, a, b, c, d, e, f):
    r2 = _r2(x, y)
    return x * sin(r2) - y * cos(r2), x * cos(r2) + y * sin(r2)

@variation
def horseshoe(x, y, a, b, c, d, e, f):
    ir = 1 / _r(x, y)
    return ir * (x - y) * (x + y), ir * 2 * x * y

@variation
def heart(x, y, a, b, c, d, e, f):
    r = _r(x, y)
    return r * sin(pi * r), -r * cos(pi * r)

@variation
def spiral(x, y, a, b, c, d, e, f):
    r = _r(x, y)
    ir = 1 / r
    t = _theta(x, y)
    return ir * cos(t) + sin(t), ir * sin(t) - cos(r)

@variation
def hyperbolic(x, y, a, b, c, d, e, f):
    r = _r(x, y)
    t = _theta(x, y)
    return sin(t) / r, r * cos(t)

# This works fine, I just don't like it
#@variation
#def diamond(x, y, a, b, c, d, e, f):
#    r = _r(x, y)
#    t = _theta(x, y)
#    return sin(t) * cos(r), cos(t) * sin(r)

@variation
def ex(x, y, a, b, c, d, e, f):
    r = _r(x, y)
    t = _theta(x, y)
    p0 = sin(t + r)
    p1 = cos(t - r)
    return r * (p0 ** 3 + p1 ** 3), r * (p0 ** 3 - p1 ** 3)

if not theanoify:
    @variation
    def julia(x, y, a, b, c, d, e, f):
        sr = sqrt(_r(x, y))
        t = _theta(x, y)
        o = _omega(x)
        return sr * cos(t / 2 + o), sr * sin(t / 2 + o)

    @variation
    def bent(x, y, a, b, c, d, e, f):
        #c1 = (x >= 0) & (y >= 0)
        if theanoify:
            c2 = T.lt(x, 0) & T.ge(y, 0)
            c3 = T.ge(x, 0) & T.lt(y, 0)
            c4 = T.lt(x, 0) & T.lt(y, 0)
            x_shifted = x * 2
            y_shifted = y / 2
            fx = T.switch(c2 | c4, x_shifted, x)
            fy = T.switch(c3 | c4, y_shifted, y)
        else:
            c2 = (x <  0) & (y >= 0)
            c3 = (x >= 0) & (y <  0)
            c4 = (x <  0) & (y <  0)
            fx = x
            fy = y
            fx[c2 | c4] = fx[c2 | c4] * 2
            fy[c3 | c4] = fy[c3 | c4] / 2
        return fx, fy

@variation
def waves(x, y, a, b, c, d, e, f):
    return x + b * sin(y / (c*c)), y + e * sin(x / (f*f))

@variation
def fisheye1(x, y, a, b, c, d, e, f):
    f = 2 / (_r(x, y) + 1)
    return f * y, f * x

