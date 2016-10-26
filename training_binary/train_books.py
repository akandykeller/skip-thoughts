import vocab
import numpy as np
import os
import re
import train

# Path to bookd data
data_path = '/data2/akeller/skip_thoughts_neon/data/books_txt_processed_full/'
dict_path = 'output_books_full/books.vocab'

save_path = 'output_books_full/model_binary_iter_{}.npz'

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


if __name__ == "__main__":
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
    train.trainer(train_sent, n_words=20000, dim=2400, batch_size=128,  reload_=False, dictionary=dict_path, saveto=save_path)  
