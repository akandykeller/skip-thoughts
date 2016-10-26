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

    # Encoder (behind)
    params = get_layer(options['encoder'])[0](options, params, prefix='encoder_b',
                                              nin=options['dim_word'], dim=options['dim'])

    # Encoder (current)
    params = get_layer(options['encoder'])[0](options, params, prefix='encoder_c',
                                              nin=options['dim_word'], dim=options['dim'])

    # Encoder (forward)
    params = get_layer(options['encoder'])[0](options, params, prefix='encoder_f',
                                              nin=options['dim_word'], dim=options['dim'])


    return params

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
    # p_n: pos sentence next
    # p_b: pos sentence behind
    # ns: negative sentences
    x = tensor.matrix('x', dtype='int64')
    x_mask = tensor.matrix('x_mask', dtype='float32')

    p_f = tensor.matrix('p_f', dtype='int64')
    p_f_mask = tensor.matrix('p_f_mask', dtype='float32')
    
    p_b = tensor.matrix('p_b', dtype='int64')
    p_b_mask = tensor.matrix('p_b_mask', dtype='float32')

    ns_list = []
    ns_masks = []
    for i in range(options['num_neg']):
        n = tensor.matrix('n_{}'.format(i), dtype='int64')
        n_mask = tensor.matrix('n_mask_{}'.format(i), dtype='float32')
        ns_list.append(n)
        ns_masks.append(n_mask)

    n_timesteps = x.shape[0]
    n_timesteps_p_b = p_b.shape[0]
    n_timesteps_p_f = p_f.shape[0]

    n_timesteps_ns = [z.shape[0] for z in ns_list]
    n_samples = x.shape[1]

    # Word embedding (source)
    source_emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps, n_samples, options['dim_word']])

    # Encoded source
    source_proj = get_layer(options['encoder'])[1](tparams, source_emb, None, options,
                                            prefix='encoder_c',
                                            mask=x_mask)
    source_proj = source_proj[0][-1]
    
     # Word embedding (Positive sentence behind)
    pos_b_emb = tparams['Wemb'][p_b.flatten()].reshape([n_timesteps_p_b, n_samples, options['dim_word']])

    # Encoded positive sentence
    pos_b_proj = get_layer(options['encoder'])[1](tparams, pos_b_emb, None, options,
                                            prefix='encoder_b',
                                            mask=p_b_mask)
    pos_b_proj = pos_b_proj[0][-1]

    # Word embedding (Positive sentence forward)
    pos_f_emb = tparams['Wemb'][p_f.flatten()].reshape([n_timesteps_p_f, n_samples, options['dim_word']])

    # Encoded forward sentence
    pos_f_proj = get_layer(options['encoder'])[1](tparams, pos_f_emb, None, options,
                                            prefix='encoder_f',
                                            mask=p_f_mask)
    pos_f_proj = pos_f_proj[0][-1]


    # Compute encodings of negatives using both foward & backward encoders
    ns_f_projs = []
    ns_b_projs = []
    for i in range(options['num_neg']):
        neg_emb = tparams['Wemb'][ns_list[i].flatten()].reshape([n_timesteps_ns[i], n_samples, options['dim_word']])

        # Encoded negative sentences w/ FORWARD encoder
        neg_f_proj = get_layer(options['encoder'])[1](tparams, neg_emb, None, options,
                                            prefix='encoder_f',
                                            mask=ns_masks[i])
        neg_f_proj = neg_f_proj[0][-1]

        ns_f_projs.append(neg_f_proj)

        # Encoded negative sentences w/ BACKWARDS encoder
        neg_b_proj = get_layer(options['encoder'])[1](tparams, neg_emb, None, options,
                                            prefix='encoder_b',
                                            mask=ns_masks[i])
        neg_b_proj = neg_b_proj[0][-1]

        ns_b_projs.append(neg_b_proj)


    # compute cosine similarities of pos & neg with source
    pos_f_cd = cosine_sim(source_proj, pos_f_proj)
    pos_b_cd = cosine_sim(source_proj, pos_b_proj)

    neg_f_cds = []
    neg_b_cds = []
    for i in range(options['num_neg']):
        neg_f_cds.append(cosine_sim(source_proj, ns_f_projs[i]))
        neg_b_cds.append(cosine_sim(source_proj, ns_b_projs[i]))

    Deltas_f = [pos_f_cd - neg_f_cd for neg_f_cd in neg_f_cds]
    Deltas_b = [pos_b_cd - neg_b_cd for neg_b_cd in neg_b_cds]

    exp_deltas_f = [tensor.exp(-options['gamma'] * d) for d in Deltas_f]
    exp_deltas_b = [tensor.exp(-options['gamma'] * d) for d in Deltas_b]

    cost_f = tensor.log(1.0 + sum(exp_deltas_f))
    cost_b = tensor.log(1.0 + sum(exp_deltas_b))

    cost = cost_f + cost_b

    return trng, x, x_mask, p_f, p_f_mask, p_b, p_b_mask, ns_list, ns_masks, opt_ret, cost

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


