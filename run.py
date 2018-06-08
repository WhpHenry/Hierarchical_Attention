import sys

from model import HAN
from utils import x_tr, y_tr, x_ts, y_ts
from utils import _SEQ_LENS, vocab_size

# han.training(x_tr, y_tr)
# han.test(x_ts, y_ts)

def main(argv, mod):
    if len(argv) <= 1 or argv[1] == '--train':
        mod.training(x_tr, y_tr)
    elif argv[1] == '--test':
        mod.test(x_ts, y_ts)
    else:
        raise Exception('ERROR: unidentified command')
        
if __name__ == '__main__':
    han = HAN(seq_lens=_SEQ_LENS, vocab_size=vocab_size)
    main(sys.argv, han)

