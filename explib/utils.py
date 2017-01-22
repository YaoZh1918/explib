from __future__ import print_function
try:
    import cPickle as pickle
except ImportError:
    import pickle

def savepkl(obj, filename):
    """> savepkl(obj, filename)
    Save given object as a pickle file.
    """
    with open(filename, 'wb') as fh:
        pickle.dump(obj, fh)


def loadpkl(filename):
    """> loadpkl(filename)
    Load pickled object from a given file.
    """
    with open(filename, 'rb') as fh:
        return pickle.load(fh)
