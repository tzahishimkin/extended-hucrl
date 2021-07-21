from path import Path
import sys
import pickle as pickle
import random
from misc.console import colorize, Message
from collections import OrderedDict
import numpy as np
import operator
from functools import reduce

sys.setrecursionlimit(50000)


def extract(x, *keys):
    if isinstance(x, (dict, lazydict)):
        return tuple(x[k] for k in keys)
    elif isinstance(x, list):
        return tuple([xi[k] for xi in x] for k in keys)
    else:
        raise NotImplementedError


def extract_dict(x, *keys):
    return {k: x[k] for k in keys if k in x}


def flatten(xs):
    return [x for y in xs for x in y]


def compact(x):
    """
    For a dictionary this removes all None values, and for a list this removes
    all None elements; otherwise it returns the input itself.
    """
    if isinstance(x, dict):
        return dict((k, v) for k, v in x.items() if v is not None)
    elif isinstance(x, list):
        return [elem for elem in x if elem is not None]
    return x


def cached_function(inputs, outputs):
    import theano
    with Message("Hashing theano fn"):
        if hasattr(outputs, '__len__'):
            hash_content = tuple(map(theano.pp, outputs))
        else:
            hash_content = theano.pp(outputs)
    cache_key = hex(hash(hash_content) & (2 ** 64 - 1))[:-1]
    cache_dir = Path('~/.hierctrl_cache')
    cache_dir = cache_dir.expanduser()
    cache_dir.mkdir_p()
    cache_file = cache_dir / ('%s.pkl' % cache_key)
    if cache_file.exists():
        with Message("unpickling"):
            with open(cache_file, "rb") as f:
                try:
                    return pickle.load(f)
                except Exception:
                    pass
    with Message("compiling"):
        fun = compile_function(inputs, outputs)
    with Message("picking"):
        with open(cache_file, "wb") as f:
            pickle.dump(fun, f, protocol=pickle.HIGHEST_PROTOCOL)
    return fun


# Immutable, lazily evaluated dict
class lazydict(object):
    def __init__(self, **kwargs):
        self._lazy_dict = kwargs
        self._dict = {}

    def __getitem__(self, key):
        if key not in self._dict:
            self._dict[key] = self._lazy_dict[key]()
        return self._dict[key]

    def __setitem__(self, i, y):
        self.set(i, y)

    def get(self, key, default=None):
        if key in self._lazy_dict:
            return self[key]
        return default

    def set(self, key, value):
        self._lazy_dict[key] = value


def iscanl(f, l, base=None):
    started = False
    for x in l:
        if base or started:
            base = f(base, x)
        else:
            base = x
        started = True
        yield base


def iscanr(f, l, base=None):
    started = False
    for x in list(l)[::-1]:
        if base or started:
            base = f(x, base)
        else:
            base = x
        started = True
        yield base


def scanl(f, l, base=None):
    return list(iscanl(f, l, base))


def scanr(f, l, base=None):
    return list(iscanr(f, l, base))


def compile_function(inputs=None, outputs=None, updates=None, givens=None, log_name=None, **kwargs):
    import theano
    if log_name:
        msg = Message("Compiling function %s" % log_name)
        msg.__enter__()
    ret = theano.function(
        inputs=inputs,
        outputs=outputs,
        updates=updates,
        givens=givens,
        on_unused_input='ignore',
        allow_input_downcast=True,
        **kwargs
    )
    if log_name:
        msg.__exit__(None, None, None)
    return ret


def new_tensor(name, ndim, dtype):
    import theano.tensor as TT
    return TT.TensorType(dtype, (False,) * ndim)(name)


def new_tensor_like(name, arr_like):
    return new_tensor(name, arr_like.ndim, arr_like.dtype)


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def is_iterable(obj):
    return isinstance(obj, str) or getattr(obj, '__iter__', False)


# cut the path for any time >= t
def truncate_path(p, t):
    return dict((k, p[k][:t]) for k in p)


def concat_paths(p1, p2):
    import numpy as np
    return dict((k1, np.concatenate([p1[k1], p2[k1]])) for k1 in list(p1.keys()) if k1 in p2)


def path_len(p):
    return len(p["states"])


def shuffled(sequence):
    deck = list(sequence)
    while len(deck):
        i = random.randint(0, len(deck) - 1)  # choose random card
        card = deck[i]  # take the card
        deck[i] = deck[-1]  # put top card in its place
        deck.pop()  # remove top card
        yield card
