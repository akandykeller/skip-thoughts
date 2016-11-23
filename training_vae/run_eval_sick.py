import tools
import eval_sick

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

if __name__=='__main__':
    embed_map = tools.load_googlenews_vectors()
    model = tools.load_model(path_to_model='output_books_full/model_full_bsz_64_M2400_iter_35000.npz', embed_map=embed_map)

    eval_sick.evaluate(model, evaltest=True)
