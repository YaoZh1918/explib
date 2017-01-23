from __future__ import print_function
import itertools
try:
    import cPickle as pickle
except ImportError:
    import pickle


class ParamsGrid(object):
    """Grid of parameters.
    """

    def __init__(self, params=dict()):
        """> __init__(self, params=dict())
        Initialize with a dict.
        Inputs:
            params: dict of string to sequence
        """
        self.independent_params = params
        self.dependent_params = list()

    def add(self, *args, **kwargs):
        """> add(self, *args, **kwargs)
        Add parameters.
        """
        if isinstance(args, dict):
            self.independent_params.update(args)
        self.independent_params.update(kwargs)

    def add_dependent(self, **kwargs):
        """> add_dependent(self, **kwargs)
        Add dependent parameters.
        """
        # check whether all values are with same length
        if len(set(map(len, kwargs.values()))) != 1:
            raise ValueError('All sequences must have the same length.')
        self.dependent_params.append(kwargs)

    def __len__(self):
        l = map(len, self.independent_params.values())
        l.extend(map(lambda x: len(x.values()[0]), self.dependent_params))
        if len(l) == 0:
            return 0
        else:
            return reduce(lambda x, y: x*y, l)

    def __str__(self):
        name = self.__class__.__name__
        items = self.independent_params.items()
        for para_dict in self.dependent_params:
            items.extend(para_dict.items())
        items = sorted(items)
        s = ', '.join(map(lambda x: '='.join(map(str, x)), items))
        return '%s(%s)' % (name, s)

    __repr__ = __str__

    def __iter__(self):
        if len(self) == 0:
            yield {}
            return
        indepen_iter = self._make_grid(self.independent_params, False)
        depen_iters = map(lambda x: self._make_grid(x, True),
                          self.dependent_params)
        for dicts in itertools.product(indepen_iter, *depen_iters):
            para = dict()
            map(para.update, dicts)
            yield para

    def _make_grid(self, para_grid, dependent):
        items = sorted(para_grid.items())
        keys, values = zip(*items)
        if dependent:
            it = itertools.izip(*values)
        else:
            it = itertools.product(*values)
        for comb in it:
            yield dict(zip(keys, comb))


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
