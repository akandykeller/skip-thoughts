import tools
import eval_sick

import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

if __name__=='__main__':
    embed_map = tools.load_googlenews_vectors()
    model = tools.load_model(embed_map)

    eval_sick.evaluate(model, evaltest=True)	

