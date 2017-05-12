import vocab
import numpy as np
import os
import re
import train

# Support for sacred!
from sacred import Experiment
ex = Experiment('skip-thought-vae')

@ex.config
def config():
    # Path to book data
    data_path = '../skip_data/books_full/'
    dict_path = 'output_books_full/books.vocab'
    save_path = 'output_books_full/model_full_VAE_bsz_256_M2400_iter_{}.npz'
    reload_path =  'output_books_full/model_full_VAE_bsz_256_M2400_iter_145000.npz'
    reload_ = True
    batch_size = 256


def clean_string(string):
    string = re.sub(r"[^A-Za-z(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def get_file_list(data_dir, file_ext):
    file_ext = file_ext if isinstance(file_ext, list) else [file_ext]
    file_names = [os.path.join(data_dir, fn) for fn in os.listdir(data_dir)
                  if any(fn.endswith(ext) for ext in file_ext)]

    return file_names


def load_txt_sent(flist_txt):
    """
    load all the senteces from a list of txt files using standard file io
    """
    all_sent = []
    for txt_file in flist_txt:
        print "Reading file: {}".format(txt_file)
        with open(txt_file, 'r') as f:
            data = f.read()
        sent = data.split('\n')
        all_sent += sent
    print "File loading complete. Cleaning..."
    #all_sent = map(clean_string, all_sent)
    return all_sent

@ex.automain
def main(data_path, dict_path, save_path, batch_size, reload_, reload_path):
    os.environ["THEANO_FLAGS"] = "floatX=float32"

    file_names = get_file_list(data_path, ['txt'])
    train_sent = load_txt_sent(file_names)

    if not os.path.exists(dict_path):
        print "Dictionary not found, recreating"
        worddict, wordcount = vocab.build_dictionary(train_sent)
        print "Built. Saving to: {}".format(dict_path)
        vocab.save_dictionary(worddict, wordcount, dict_path)
    else:
        print "Found dictionary at {}... Loading...".format(dict_path)
        worddict = vocab.load_dictionary(dict_path)
   
    print "Beginning Training..." 
    train.trainer(train_sent, batch_size=batch_size,  reload_=reload_, dictionary=dict_path, saveto=save_path, reload_path=reload_path, saveFreq=10000)  
