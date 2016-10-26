import numpy
import copy

class HomogeneousData():

    def __init__(self, data, batch_size=128, num_neg=4, maxlen=None):
        self.data = data
        self.batch_size = batch_size
        self.maxlen = maxlen
        self.num_neg = num_neg

        self.prepare()
        self.reset()

    def prepare(self):
        self.caps = self.data[0]
        self.pos = self.data[1]
        self.pos2 = self.data[2]

        # find the unique lengths
        self.lengths = [len(cc.split()) for cc in self.caps]
        self.len_unique = numpy.unique(self.lengths)
        # remove any overly long sentences
        if self.maxlen:
            self.len_unique = [ll for ll in self.len_unique if ll <= self.maxlen]

        # indices of unique lengths
        self.len_indices = dict()
        self.len_counts = dict()
        for ll in self.len_unique:
            self.len_indices[ll] = numpy.where(self.lengths == ll)[0]
            self.len_counts[ll] = len(self.len_indices[ll])

        # current counter
        self.len_curr_counts = copy.copy(self.len_counts)

    def reset(self):
        self.len_curr_counts = copy.copy(self.len_counts)
        self.len_unique = numpy.random.permutation(self.len_unique)
        self.len_indices_pos = dict()
        for ll in self.len_unique:
            self.len_indices_pos[ll] = 0
            self.len_indices[ll] = numpy.random.permutation(self.len_indices[ll])
        self.len_idx = -1

    def next(self):
        count = 0
        while True:
            self.len_idx = numpy.mod(self.len_idx+1, len(self.len_unique))
            if self.len_curr_counts[self.len_unique[self.len_idx]] > 0:
                break
            count += 1
            if count >= len(self.len_unique):
                break
        if count >= len(self.len_unique):
            self.reset()
            raise StopIteration()

        # get the batch size
        curr_batch_size = numpy.minimum(self.batch_size, self.len_curr_counts[self.len_unique[self.len_idx]])
        curr_pos = self.len_indices_pos[self.len_unique[self.len_idx]]
        # get the indices for the current batch
        curr_indices = self.len_indices[self.len_unique[self.len_idx]][curr_pos:curr_pos+curr_batch_size]
        self.len_indices_pos[self.len_unique[self.len_idx]] += curr_batch_size
        self.len_curr_counts[self.len_unique[self.len_idx]] -= curr_batch_size

        # 'pos' corresponds to the after sentences, 'pos2' is before sentences
        caps = [self.caps[ii] for ii in curr_indices]
        pos = [self.pos[ii] for ii in curr_indices]
        pos2 = [self.pos2[ii] for ii in curr_indices]

        # we need num_neg randomly picked negative sentences for each minibatch. 
        try:
            non_pos_indicies = numpy.r_[0:curr_pos, curr_pos+curr_batch_size:len(self.len_indices[self.len_unique[self.len_idx]])]
            neg_indicies_rand = numpy.random.choice(self.len_indices[self.len_unique[self.len_idx]][non_pos_indicies],
                                                 size=self.num_neg)
        except ValueError:
            non_pos_indicies = list(curr_indices + 1) + list(curr_indices - 1) + list(curr_indices)
            neg_indicies = list(set(xrange(len(self.caps))) - set(non_pos_indicies))
            neg_indicies_rand = numpy.random.choice(neg_indicies, size=self.num_neg)

        negs = [self.caps[ii] for ii in neg_indicies_rand]

        return caps, pos, pos2, negs

    def __iter__(self):
        return self

def prepare_data(seqs_x, seqs_p1, seqs_p2, ns, worddict, maxlen=None, n_words=20000):
    """
    Put the data into format useable by the model

    (x -> (maxlen_x, n_samples)
     p1 -> (maxlen_p1, n_samples)
     p2 -> (maxlen_p2, n_samples),
     ns -> 5 * (neg_sentence, mask) * n_samples?
            (num_neg, maxlen_ns, n_samples)   )

    """
    # Iterate over words in each sentence of the minibatch
    seqsX = []
    seqsP1 = []
    seqsP2 = []
    for cc in seqs_x:
        seqsX.append([worddict[w] if worddict[w] < n_words else 1 for w in cc.split()])
    for cc in seqs_p1:
        seqsP1.append([worddict[w] if worddict[w] < n_words else 1 for w in cc.split()])
    for cc in seqs_p2:
        seqsP2.append([worddict[w] if worddict[w] < n_words else 1 for w in cc.split()])
    
    # Tokenize each of the negative sentences 
    seqsN = []
    for nn in ns:
        seqsN.append([worddict[w] if worddict[w] < n_words else 1 for w in nn.split()])

    seqs_x = seqsX
    seqs_p1 = seqsP1
    seqs_p2 = seqsP2
    seqs_n = seqsN

    lengths_x = [len(s) for s in seqs_x]
    lengths_p1 = [len(s) for s in seqs_p1]
    lengths_p2 = [len(s) for s in seqs_p2]

    # lengths_n is only num_neg long, since each batch of negatives have same length
    lengths_n = [len(n) for n in seqs_n]

    if maxlen != None:
        # Before anything, check length of negative samples, and bail if any is too long
        if not numpy.alltrue([l_n < maxlen for l_n in lengths_n]):
            return None, None, None, None, None, None, None, None

        new_seqs_x = []
        new_seqs_p1 = []
        new_seqs_p2 = []

        new_lengths_x = []
        new_lengths_p1 = []
        new_lengths_p2 = []

        for l_x, s_x, l_p1, s_p1, l_p2, s_p2 in zip(lengths_x, seqs_x, lengths_p1, seqs_p1, lengths_p2, seqs_p2):
            if l_x < maxlen and l_p1 < maxlen and l_p2 < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
                new_seqs_p1.append(s_p1)
                new_lengths_p1.append(l_p1)
                new_seqs_p2.append(s_p2)
                new_lengths_p2.append(l_p2)

        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
        lengths_p1 = new_lengths_p1
        seqs_p1 = new_seqs_p1
        lengths_p2 = new_lengths_p2
        seqs_p2 = new_seqs_p2

        if len(lengths_x) < 1 or len(lengths_p1) < 1 or len(lengths_p2) < 1:
            return None, None, None, None, None, None, None, None

    n_samples = len(seqs_x)
    maxlen_x = numpy.max(lengths_x) + 1
    maxlen_p1 = numpy.max(lengths_p1) + 1
    maxlen_p2 = numpy.max(lengths_p2) + 1

    x = numpy.zeros((maxlen_x, n_samples)).astype('int64')
    p1 = numpy.zeros((maxlen_p1, n_samples)).astype('int64')
    p2 = numpy.zeros((maxlen_p2, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen_x, n_samples)).astype('float32')
    p1_mask = numpy.zeros((maxlen_p1, n_samples)).astype('float32')
    p2_mask = numpy.zeros((maxlen_p2, n_samples)).astype('float32')

    # Copy each negative sentence n_samples times to share over the minibatch
    ns_list = []
    ns_masks = []
    for n_idx in range(len(ns)):
        n = numpy.zeros((lengths_n[n_idx] + 1, n_samples)).astype('int64')
        n_mask = numpy.zeros((lengths_n[n_idx] + 1, n_samples)).astype('float32')

        for b_idx in range(n_samples):
            n[:lengths_n[n_idx], b_idx] = seqs_n[n_idx]
            n_mask[:lengths_n[n_idx]+1, b_idx] = 1.

        ns_list.append(n)
        ns_masks.append(n_mask)

    for idx, [s_x, s_p1, s_p2] in enumerate(zip(seqs_x,seqs_p1,seqs_p2)):
        x[:lengths_x[idx],idx] = s_x
        x_mask[:lengths_x[idx]+1,idx] = 1.
        p1[:lengths_p1[idx],idx] = s_p1
        p1_mask[:lengths_p1[idx]+1,idx] = 1.
        p2[:lengths_p2[idx],idx] = s_p2
        p2_mask[:lengths_p2[idx]+1,idx] = 1.

    return x, x_mask, p1, p1_mask, p2, p2_mask, ns_list, ns_masks

def grouper(text):
    """
    Group text into triplets
    """
    source = text[1:][:-1]
    forward = text[2:]
    backward = text[:-2]
    X = (source, forward, backward)
    return X


