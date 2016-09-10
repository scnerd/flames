default_condense_args = {
    'xmin': -1,
    'xmax': 1,
    'ymin': -1,
    'ymax': 1,
    'xres': 1440,
    'yres': 1080,
    'gamma': 2.0,
    'vibrancy': 0.0,
    'supersample': 2,
    'saturation': None, # or a real value
    'contrast': None, # or a real value
    'threads_per_block': 100,
    'length': 100,
    'n': 1000000,
    'num_chans': 4,
}
make_get = lambda d: lambda k: d[k] if k in d else default_condense_args[k]
theano_dtype = 'float32'