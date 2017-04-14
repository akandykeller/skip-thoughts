"""
Layers for skip-thoughts

To add a new layer:
1) Add layer names to the 'layers' dictionary below
2) Implement param_init and feedforward functions
3) In the trainer function, replace 'encoder' or 'decoder' with your layer name

"""
import theano
import theano.tensor as tensor

import numpy

from utils import _p, ortho_weight, norm_weight, tanh, linear, relu

# layers: 'name': ('parameter initializer', 'feedforward')
layers = {'ff': ('param_init_fflayer', 'fflayer'),
          'gru': ('param_init_gru', 'gru_layer'),
          'vae': ('param_init_vae', 'vae_layer')
          }

def get_layer(name):
    """
    Return param init and feedforward functions for the given layer name
    """
    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))

# Feedforward layer
def param_init_fflayer(options, params, prefix='ff', nin=None, nout=None, ortho=True):
    """
    Affine transformation + point-wise nonlinearity
    """
    if nin == None:
        nin = options['dim_proj']
    if nout == None:
        nout = options['dim_proj']
    params[_p(prefix,'W')] = norm_weight(nin, nout, ortho=ortho)
    params[_p(prefix,'b')] = numpy.zeros((nout,)).astype('float32')

    return params

def fflayer(tparams, state_below, options, prefix='rconv', activ='lambda x: tensor.tanh(x)', **kwargs):
    """
    Feedforward pass
    """
    return eval(activ)(tensor.dot(state_below, tparams[_p(prefix,'W')])+tparams[_p(prefix,'b')])


# Vae Layer
def param_init_vae(options, params, prefix='vae', nhid=None, nlatent=None, ndim=None, ortho=True):
    """
    Variational Layer
    """
    if nhid == None:
        nhid = options['vae_nhid']
    if nlatent == None:
        nlatent = options['vae_nlatent']
    if ndim == None:
        ndim = options['dim']

    # VAE encoder part
    # params[_p(prefix,'W_xh')] = norm_weight(ndim, nhid, ortho=ortho)
    # params[_p(prefix,'b_xh')] = numpy.zeros((nhid,)).astype('float32')

    params[_p(prefix,'W_hmu')] = norm_weight(nhid, nlatent, ortho=ortho)
    params[_p(prefix,'b_hmu')] = numpy.zeros((nlatent,)).astype('float32')

    params[_p(prefix,'W_hsigma')] = norm_weight(nhid, nlatent, ortho=ortho)
    params[_p(prefix,'b_hsigma')] = numpy.zeros((nlatent,)).astype('float32')

    params[_p(prefix,'W_zh')] = norm_weight(nlatent, nhid, ortho=ortho)
    params[_p(prefix,'b_zh')] = numpy.zeros((nhid,)).astype('float32')

    # params[_p(prefix,'W_hx')] = norm_weight(nhid, ndim, ortho=ortho)
    # params[_p(prefix,'b_hx')] = numpy.zeros((ndim,)).astype('float32')

    return params

def vae_layer(tparams, state_below, options, prefix='vae', **kwargs):
    """
    VAE
    """
    seed = 42

    # Encode 
    # h_encoder = relu(tensor.dot(state_below, tparams[_p(prefix, 'W_xh')]) + tparams[_p(prefix, 'b_xh')])

    mu = tensor.dot(state_below, tparams[_p(prefix, 'W_hmu')]) + tparams[_p(prefix, 'b_hmu')]
    log_sigma = tensor.dot(state_below, tparams[_p(prefix, 'W_hsigma')]) + tparams[_p(prefix, 'b_hsigma')]
    
    if "gpu" in theano.config.device:
        srng = theano.sandbox.cuda.rng_curand.CURAND_RandomStreams(seed=seed)
    else:
        srng = tensor.shared_randomstreams.RandomStreams(seed=seed)

    eps = srng.normal(mu.shape)

    # Reparametrize
    z = mu + tensor.exp(0.5 * log_sigma) * eps

    # Decode 
    h_decoder = tensor.tanh(tensor.dot(z, tparams[_p(prefix, 'W_zh')]) + tparams[_p(prefix, 'b_zh')])

    # h_out = tensor.dot(h_decoder, tparams[_p(prefix, 'W_hx')]) + tparams[_p(prefix, 'b_hx')]

    # Returned output is same shape as RNN decoder hidden state, used to initalize decoders 
    return h_decoder, mu, log_sigma


# GRU layer
def param_init_gru(options, params, prefix='gru', nin=None, dim=None):
    """
    Gated Recurrent Unit (GRU)
    """
    if nin == None:
        nin = options['dim_proj']
    if dim == None:
        dim = options['dim_proj']
    W = numpy.concatenate([norm_weight(nin,dim),
                           norm_weight(nin,dim)], axis=1)
    params[_p(prefix,'W')] = W
    params[_p(prefix,'b')] = numpy.zeros((2 * dim,)).astype('float32')
    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
    params[_p(prefix,'U')] = U

    Wx = norm_weight(nin, dim)
    params[_p(prefix,'Wx')] = Wx
    Ux = ortho_weight(dim)
    params[_p(prefix,'Ux')] = Ux
    params[_p(prefix,'bx')] = numpy.zeros((dim,)).astype('float32')

    return params

def gru_layer(tparams, state_below, init_state, options, prefix='gru', mask=None, **kwargs):
    """
    Feedforward pass through GRU
    """
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    dim = tparams[_p(prefix,'Ux')].shape[1]

    if init_state == None:
        init_state = tensor.alloc(0., n_samples, dim)

    if mask == None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) + tparams[_p(prefix, 'bx')]
    U = tparams[_p(prefix, 'U')]
    Ux = tparams[_p(prefix, 'Ux')]

    def _step_slice(m_, x_, xx_, h_, U, Ux):
        preact = tensor.dot(h_, U)
        preact += x_

        r = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        u = tensor.nnet.sigmoid(_slice(preact, 1, dim))

        preactx = tensor.dot(h_, Ux)
        preactx = preactx * r
        preactx = preactx + xx_

        h = tensor.tanh(preactx)

        h = u * h_ + (1. - u) * h
        h = m_[:,None] * h + (1. - m_)[:,None] * h_

        return h

    seqs = [mask, state_below_, state_belowx]
    _step = _step_slice

    rval, updates = theano.scan(_step,
                                sequences=seqs,
                                outputs_info = [init_state],
                                non_sequences = [tparams[_p(prefix, 'U')],
                                                 tparams[_p(prefix, 'Ux')]],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps,
                                profile=False,
                                strict=True)
    rval = [rval]
    return rval


