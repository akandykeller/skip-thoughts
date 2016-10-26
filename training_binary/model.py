"""
Model specification
"""
import theano
import theano.tensor as tensor
import numpy

from collections import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from utils import _p, ortho_weight, norm_weight, tanh
from layers import get_layer, param_init_fflayer, fflayer, param_init_gru, gru_layer

def init_params(options):
    """
    Initialize all parameters
    """
    params = OrderedDict()

    # Word embedding
    params['Wemb'] = norm_weight(options['n_words'], options['dim_word'])

    # Encoder
    params = get_layer(options['encoder'])[0](options, params, prefix='encoder',
                                              nin=options['dim_word'], dim=options['dim'])

    return params

# def cosine_sim(a, b):
#     """
#     Return cosine similarity measure of two vectors a and b
#     """
#     num = tensor.sum(a * b)
#     denom = tensor.sqrt(tensor.sum(a ** 2) * tensor.sum(b ** 2))
#     return num / denom

def _squared_magnitude(x):
    return tensor.sqr(x).sum()

def _magnitude(x):
    return tensor.sqrt(tensor.maximum(_squared_magnitude(x), numpy.finfo(x.dtype).tiny))

def cosine_sim(x, y):
    return tensor.clip((1 - (x * y).sum() / (_magnitude(x) * _magnitude(y))) / 2, 0, 1)


def build_model(tparams, options):
    """
    Computation graph for the model
    """
    opt_ret = dict()

    trng = RandomStreams(1234)

    # description string: #words x #samples
    # x: current sentence
    # p: pos sentence 
    # ns: negative sentences
    x = tensor.matrix('x', dtype='int64')
    x_mask = tensor.matrix('x_mask', dtype='float32')

    p = tensor.matrix('p', dtype='int64')
    p_mask = tensor.matrix('p_mask', dtype='float32')
    
    ns_list = []
    ns_masks = []
    for i in range(options['num_neg']):
        n = tensor.matrix('n_{}'.format(i), dtype='int64')
        n_mask = tensor.matrix('n_mask_{}'.format(i), dtype='float32')
        ns_list.append(n)
        ns_masks.append(n_mask)

    n_timesteps = x.shape[0]
    n_timesteps_p = p.shape[0]
    n_timesteps_ns = [z.shape[0] for z in ns_list]
    n_samples = x.shape[1]

    # Word embedding (source)
    source_emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps, n_samples, options['dim_word']])

    # Encoded source
    source_proj = get_layer(options['encoder'])[1](tparams, source_emb, None, options,
                                            prefix='encoder',
                                            mask=x_mask)
    source_proj = source_proj[0][-1]
    

     # Word embedding (Positive sentence)
    pos_emb = tparams['Wemb'][p.flatten()].reshape([n_timesteps_p, n_samples, options['dim_word']])

    # Encoded positive sentence
    pos_proj = get_layer(options['encoder'])[1](tparams, pos_emb, None, options,
                                            prefix='encoder',
                                            mask=p_mask)
    pos_proj = pos_proj[0][-1]


    ns_projs = []
    for i in range(options['num_neg']):
        neg_emb = tparams['Wemb'][ns_list[i].flatten()].reshape([n_timesteps_ns[i], n_samples, options['dim_word']])

        # Encoded negative sentences
        neg_proj = get_layer(options['encoder'])[1](tparams, neg_emb, None, options,
                                            prefix='encoder',
                                            mask=ns_masks[i])
        neg_proj = neg_proj[0][-1]

        ns_projs.append(neg_proj)


    # compute cosine similarities of pos & neg with source
    pos_cd = cosine_sim(source_proj, pos_proj)
    neg_cds = []

    for i in range(options['num_neg']):
        neg_cds.append(cosine_sim(source_proj, ns_projs[i]))


    Deltas = [pos_cd - neg_cd for neg_cd in neg_cds]

    exp_deltas = [tensor.exp(-options['gamma'] * d) for d in Deltas]

    cost = tensor.log(1.0 + sum(exp_deltas))

    return trng, x, x_mask, p, p_mask, ns_list, ns_masks, opt_ret, cost

def build_encoder(tparams, options):
    """
    Computation graph, encoder only
    """
    opt_ret = dict()

    trng = RandomStreams(1234)

    # description string: #words x #samples
    x = tensor.matrix('x', dtype='int64')
    x_mask = tensor.matrix('x_mask', dtype='float32')

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    # word embedding (source)
    emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps, n_samples, options['dim_word']])

    # encoder
    proj = get_layer(options['encoder'])[1](tparams, emb, None, options,
                                            prefix='encoder',
                                            mask=x_mask)
    ctx = proj[0][-1]

    return trng, x, x_mask, ctx, emb

def build_encoder_w2v(tparams, options):
    """
    Computation graph for encoder, given pre-trained word embeddings
    """
    opt_ret = dict()

    trng = RandomStreams(1234)

    # word embedding (source)
    embedding = tensor.tensor3('embedding', dtype='float32')
    x_mask = tensor.matrix('x_mask', dtype='float32')

    # encoder
    proj = get_layer(options['encoder'])[1](tparams, embedding, None, options,
                                            prefix='encoder',
                                            mask=x_mask)
    ctx = proj[0][-1]

    return trng, embedding, x_mask, ctx


