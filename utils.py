import numpy as np
from keras.datasets import lmdb

_NUM_WORDS = 10000
_INDEX_FROM = 3
_SEQ_LENS = 250

def load_data(path='data/imdb.npz', num_words=None, skip_top=0,
              seed=113, start_char=1, oov_char=2, index_from=3):
    '''
        copy from keras.datasets.imdb.load_data
        if you cannot download by imdb.load_data, like me
        you can download from: 
        https://s3.amazonaws.com/text-datasets/imdb.npz
        put imdb.npz into ./data/
        then take load_data function here insteads imdb.load_data
    '''
    with np.load(path) as f:
        x_train, labels_train = f['x_train'], f['y_train']
        x_test, labels_test = f['x_test'], f['y_test']

    np.random.seed(seed)
    indices = np.arange(len(x_train))
    np.random.shuffle(indices)
    x_train = x_train[indices]
    labels_train = labels_train[indices]

    indices = np.arange(len(x_test))
    np.random.shuffle(indices)
    x_test = x_test[indices]
    labels_test = labels_test[indices]

    xs = np.concatenate([x_train, x_test])
    labels = np.concatenate([labels_train, labels_test])

    if start_char is not None:
        xs = [[start_char] + [w + index_from for w in x] for x in xs]
    elif index_from:
        xs = [[w + index_from for w in x] for x in xs]

    if not num_words:
        num_words = max([max(x) for x in xs])

    # by convention, use 2 as OOV word
    # reserve 'index_from' (=3 by default) characters:
    # 0 (padding), 1 (start), 2 (OOV)
    if oov_char is not None:
        xs = [[w if (skip_top <= w < num_words) else oov_char for w in x] for x in xs]
    else:
        xs = [[w for w in x if skip_top <= w < num_words] for x in xs]

    idx = len(x_train)
    x_train, y_train = np.array(xs[:idx]), np.array(labels[:idx])
    x_test, y_test = np.array(xs[idx:]), np.array(labels[idx:])

    return (x_train, y_train), (x_test, y_test)

def zero_pad(X, seq_len=_SEQ_LENS):
    return np.array([x[:seq_len - 1] + [0] * max(seq_len - len(x), 1) for x in X])

def get_vob_size(source):
    return max([max(x) for x in source]) + 1  # plus the 0th word 

def fit_in_vob(source, vob_size):
    return [[w for w in x if w < vob_size] for x in source]

    
(x_tr, y_tr), (x_ts, y_ts) = load_data(num_words=_NUM_WORDS, index_from=_INDEX_FROM)

vocab_size = get_vob_size(x_tr)
x_tr = zero_pad(x_tr)
x_ts = fit_in_vob(x_ts, vocab_size)
x_ts = zero_pad(x_ts)
