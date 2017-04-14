"""
Main trainer function
"""
import theano
import theano.tensor as tensor

import cPickle as pkl
import numpy
import copy

import os
import warnings
import sys
import time
from time import gmtime, strftime

import homogeneous_data

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import numpy as np
from utils import *
from layers import get_layer, param_init_fflayer, fflayer, param_init_gru, gru_layer
from optim import adam
from model import init_params, build_model
from vocab import load_dictionary
import tools
import eval_sick

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# main trainer
def trainer(X, 
            dim_word=620, # word vector dimensionality
            dim=2400, # the number of GRU units
            encoder='gru',
            vae='vae',
            vae_nhid=300,
            vae_nlatent=20,
            decoder='gru',
            max_epochs=5,
            dispFreq=1,
            decay_c=0.,
            grad_clip=5.,
            n_words=20000,
            maxlen_w=30,
            optimizer='adam',
            batch_size = 64,
            saveto='/u/rkiros/research/semhash/models/toy.npz',
            reload_path='output_books_full/model_full_bsz_64_M2400_iter_35000.npz',
            dictionary='/ais/gobi3/u/rkiros/bookgen/book_dictionary_large.pkl',
            saveFreq=5000,
            reload_=False,
            SICK_eval=False):

    # Model options
    model_options = {}
    model_options['dim_word'] = dim_word
    model_options['dim'] = dim
    model_options['encoder'] = encoder
    model_options['vae'] = vae
    model_options['vae_nhid'] = vae_nhid
    model_options['vae_nlatent'] = vae_nlatent
    model_options['decoder'] = decoder 
    model_options['max_epochs'] = max_epochs
    model_options['dispFreq'] = dispFreq
    model_options['decay_c'] = decay_c
    model_options['grad_clip'] = grad_clip
    model_options['n_words'] = n_words
    model_options['maxlen_w'] = maxlen_w
    model_options['optimizer'] = optimizer
    model_options['batch_size'] = batch_size
    model_options['saveto'] = saveto
    model_options['dictionary'] = dictionary
    model_options['saveFreq'] = saveFreq
    model_options['reload_'] = reload_
    model_options['reload_path'] = reload_path

    print model_options

    # reload options
    if reload_ and os.path.exists(reload_path):
        print 'reloading...' + reload_path
        with open('%s.pkl'%reload_path, 'rb') as f:
            models_options = pkl.load(f)
        
        
        reload_idx = int(reload_path.split('_')[-1].split('.')[0])

    # load dictionary
    print 'Loading dictionary...'
    worddict = load_dictionary(dictionary)

    # Inverse dictionary
    word_idict = dict()
    for kk, vv in worddict.iteritems():
        word_idict[vv] = kk
    word_idict[0] = '<eos>'
    word_idict[1] = 'UNK'

    print 'Building model'
    params = init_params(model_options)
    # reload parameters
    if reload_ and os.path.exists(reload_path):
        params = load_params(reload_path, params)

    tparams = init_tparams(params)

    trng, x, x_mask, y, y_mask, z, z_mask, \
	  KL_w, KL_val, \
          opt_ret, \
          cost = \
          build_model(tparams, model_options)
    inps = [x, x_mask, y, y_mask, z, z_mask, KL_w]

    # before any regularizer
    print 'Building f_log_probs...',
    f_log_probs = theano.function(inps, cost, profile=False)
    print 'Done'

    print 'Building f_KL_val'
    f_KL_val = theano.function([x, x_mask], KL_val, profile=False)

    # weight decay, if applicable
    if decay_c > 0.:
        decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in tparams.iteritems():
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    # after any regularizer
    print 'Building f_cost...',
    f_cost = theano.function(inps, cost, profile=False)
    print 'Done'

    print 'Done'
    print 'Building f_grad...',
    grads = tensor.grad(cost, wrt=itemlist(tparams))
    f_grad_norm = theano.function(inps, [(g**2).sum() for g in grads], profile=False)
    f_weight_norm = theano.function([], [(t**2).sum() for k,t in tparams.iteritems()], profile=False)

    if grad_clip > 0.:
        g2 = 0.
        for g in grads:
            g2 += (g**2).sum()
        new_grads = []
        for g in grads:
            new_grads.append(tensor.switch(g2 > (grad_clip**2),
                                           g / tensor.sqrt(g2) * grad_clip,
                                           g))
        grads = new_grads

    lr = tensor.scalar(name='lr')
    print 'Building optimizers...',
    # (compute gradients), (updates parameters)
    f_grad_shared, f_update = eval(optimizer)(lr, tparams, grads, inps, cost)

    print "Loading embed map"

    print 'Optimization'

    # Each sentence in the minibatch have same length (for encoder)
    trainX = homogeneous_data.grouper(X)
    train_iter = homogeneous_data.HomogeneousData(trainX, batch_size=batch_size, maxlen=maxlen_w)

    if not reload_:
        uidx = 0
    else:
        uidx = reload_idx
    lrate = 0.005
    for eidx in xrange(max_epochs):
        n_samples = 0

        print 'Epoch ', eidx

        for x, y, z in train_iter:
            n_samples += len(x)
            uidx += 1

            x, x_mask, y, y_mask, z, z_mask = homogeneous_data.prepare_data(x, y, z, worddict, maxlen=maxlen_w, n_words=n_words)

            if x == None:
                print 'Minibatch with zero sample under length ', maxlen_w
                uidx -= 1
                continue

            KL_w = ((np.tanh(1.0/2000.0*(uidx - 5000.0)) + 1.0) / 2.0).astype(np.float32)

            ud_start = time.time()
            cost = f_grad_shared(x, x_mask, y, y_mask, z, z_mask, KL_w)
	    KL_val = f_KL_val(x, x_mask)
            f_update(lrate)
            ud = time.time() - ud_start

            if numpy.isnan(cost) or numpy.isinf(cost):
                print 'NaN detected'
                return 1., 1., 1.

            if numpy.mod(uidx, dispFreq) == 0:
                print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost, 'UD ', ud, 'KL_val', KL_val.mean(), 'KL_w', KL_w

            if numpy.mod(uidx, saveFreq) == 0:
                print 'Saving...',
                
                saveto_iternum = saveto.format(uidx)

                params = unzip(tparams)
                numpy.savez(saveto_iternum, history_errs=[], **params)
                pkl.dump(model_options, open('%s.pkl'%saveto_iternum, 'wb'))
                print 'Done'

                if SICK_eval:
                    print "Evaluating SICK Test performance"
                    embed_map = tools.load_googlenews_vectors()
                    model = tools.load_model(path_to_model=saveto_iternum, embed_map=embed_map)
                    yhat, pr, sr, mse = eval_sick.evaluate(model, evaltest=True)
                    
                    del(model)        
                    del(embed_map) 
                    print pr, sr, mse

                    res_save_file = saveto.format('ALL').split('.')[0] + '_SICK_EVAL.txt'
                    with open(res_save_file, 'a') as rsf:
                        cur_time = strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())
                        rsf.write('\n \n{}'.format(cur_time))
                        rsf.write('\n{}, {}, {}, {}'.format(uidx, pr, sr, mse))
                    print "Done" 

        print 'Seen %d samples'%n_samples

if __name__ == '__main__':
    pass


