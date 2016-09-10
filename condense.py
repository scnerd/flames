import numpy as np
import theano
from args import *

def make_condense(**condensed_args):
    '''
    Take a 2D histogram of where points have landed
    This is actually a wrapper function to store the parameters for the histogram, just so
    we're not re-extracting parameters every iteration (not really a big deal, but still)
    '''
    get = make_get(condensed_args)
    xmin = get('xmin')
    xmax = get('xmax')
    ymin = get('ymin')
    ymax = get('ymax')
    xres = get('xres')
    yres = get('yres')
    supersample = get('supersample')
    def condense(x, y, c):
        return np.stack([np.histogram2d(x, y, weights=c[:,channel],
                                        bins=[yres*supersample, xres*supersample],
                                        range=[[ymin, ymax], [xmin, xmax]])[0]
                         for channel in range(4)],
                        axis=2)
    return condense

from pycuda.compiler import SourceModule
import theano.misc.pycuda_init
import theano.sandbox.cuda as cuda

class PyCUDAHist2D(theano.Op):
    __props__ = ('n', 'xres', 'yres', 'xmin', 'xptp', 'ymin', 'yptp', 'num_chans', 'length')

    def __init__(self, **condensed_args):
        get = make_get(condensed_args)
        self.args = {
            'n': get('n'),
            'xres': get('xres') * get('supersample'),
            'yres': get('yres') * get('supersample'),
            'xmin': float(get('xmin')),
            'xptp': float(get('xmax') - get('xmin')),
            'ymin': float(get('ymin')),
            'yptp': float(get('ymax') - get('ymin')),
            'num_chans': condensed_args.get('num_chans', 4),
            'length': condensed_args.get('length', np.min([1000, int(np.sqrt(get('n')))])),
        }
        self.n = self.args['n']
        self.xres = self.args['xres']
        self.yres = self.args['yres']
        self.xmin = self.args['xmin']
        self.xptp = self.args['xptp']
        self.ymin = self.args['ymin']
        self.yptp = self.args['yptp']
        self.num_chans = self.args['num_chans']
        self.length = self.args['length']


    def make_node(self, x, y, w):
        x = cuda.basic_ops.gpu_contiguous(cuda.basic_ops.as_cuda_ndarray_variable(x))
        y = cuda.basic_ops.gpu_contiguous(cuda.basic_ops.as_cuda_ndarray_variable(y))
        w = cuda.basic_ops.gpu_contiguous(cuda.basic_ops.as_cuda_ndarray_variable(w))
        out_t = cuda.CudaNdarrayType((False, False, False, False), dtype='float32')()
        assert x.dtype == 'float32'
        assert y.dtype == 'float32'
        assert w.dtype == 'float32'
        return theano.Apply(self, [x, y, w], [out_t])

    # Based on
    #https://devblogs.nvidia.com/parallelforall/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/
    def make_thunk(self, node, storage_map, _, _2):

        # Creates a histogram in BLOCKS_X pieces, which then need to be added together
        code = '''
        __global__ void histogram2d(const float *in_x, const float *in_y, const float *in_w, float *out) {{
            int start = (blockIdx.x * blockDim.x + threadIdx.x) * {length};

            float *block_out = &out[{xres} * {yres} * {num_chans} * blockIdx.x];
            //float *block_out = out;

            for(int i = 0; i < {length}; i++) {{
                float x = in_x[start + i];
                float y = in_y[start + i];
                int w_idx = (start + i) * {num_chans};

                int xbin = (int) (((x - {xmin}) / {xptp}) * {xres});
                int ybin = (int) (((y - {ymin}) / {yptp}) * {yres});

                if (0 <= xbin && xbin < {xres} && 0 <= ybin && ybin < {yres}) {{
                    for(int c = 0; c < {num_chans}; c++) {{
                        atomicAdd(&block_out[(ybin * {xres} + xbin) * {num_chans} + c], in_w[w_idx + c]);
                    }}
                }}
            }}
        }}
        '''.format(**self.args)
        mod = SourceModule(code)
        cuda_hist = mod.get_function('histogram2d')

        inputs = [storage_map[v] for v in node.inputs]
        outputs = [storage_map[v] for v in node.outputs]

        _x, _y, _w = inputs
        (_out,) = outputs

        def run_hist():
            x = _x[0]
            y = _y[0]
            w = _w[0]
            n = x.size
            xres = self.args['xres']
            yres = self.args['yres']
            num_chans = self.args['num_chans']
            length = self.args['length']
            num_blocks = 2
            threads_per_block = int(n / length / num_blocks)
            if _out[0] is None or _out[0].shape != (num_blocks, yres, xres, num_chans):
                _out[0] = cuda.CudaNdarray.zeros((num_blocks, yres, xres, num_chans))
            cuda_hist(x, y, w, _out[0], block=(threads_per_block,1,1), grid=(num_blocks,1))

        return run_hist#, zeros((num_blocks, yres, xres, num_chans), 'float32')