from __future__ import print_function
import itertools
import numpy as np
import pandas as pd
from time import strftime
import logging
import os
try:
    import cPickle as pickle
    from cPickle import UnpicklingError
except ImportError:
    import pickle
    from pickle import UnpicklingError


logger = logging.getLogger(__name__)


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


def parse_result(data, ops=['mean', 'std', 'max', 'min']):
    """> parse_result(data, ops=['mean', 'std', 'max', 'min'])
    Parse a single experiment result.
    """
    line = dict()
    ops_dict = dict(mean=np.mean, std=np.std, max=np.max, min=np.min)
    # Options
    for prefix, opt in sorted(data['Options'].items()):
        if prefix == 'metrics':
            continue
        for k, v in opt.__dict__.iteritems():
            line['_'.join([prefix, k])] = v
    # Metrics
    def make_name(opt):
        items = filter(lambda x: x[0] != '_name', opt.__dict__.items())
        s = ','.join(map(lambda x: '%s=%s' % x, items))
        return '%s(%s)' % (opt._name, s)
    for m_opt, arr in data['Metrics']:
        prefix = make_name(m_opt)
        for op in ops:
            line['_'.join([prefix, op])] = ops_dict[op](arr)
    # Others
    if isinstance(data['Others'], dict):
        for k, v in data['Others'].iteritems():
            line['_'.join(['others', k])] = v
    return line



def merge_result(dir_name, ops=['mean', 'std', 'max', 'min']):
    """> merge_result(dir_name, ops=['mean', 'std', 'max', 'min'])
    Merge all results under a directory.
    Return a pandas.DataFrame and a pandas.Series."""
    all_data = list()
    for c_dir, _, file_list in os.walk(dir_name):
        for c_file in file_list:
            try:
                filename = os.path.join(c_dir, c_file)
                data = loadpkl(filename)
                all_data.append(parse_result(data, ops))
            except (IOError, EOFError, UnpicklingError), e:
                logger.error('Load %s failed, pass' % filename)
    df = pd.DataFrame(all_data)
    # define cmp method
    def df_cmp(x, y):
        def get_level(s):
            pos = map(s.startswith, ['dataset', 'model', 'setting', ''])
            return int(np.where(pos)[0][0])
        lx, ly = map(get_level, [x, y])
        if lx == ly:
            a, b = map(lambda x: x.endswith('name'), [x, y])
            if a == b: return x < y
            else: return b - a
        else: return lx - ly
    cols = sorted(df.columns, cmp=df_cmp)
    df = df[cols]
    # drop duplicates
    keep_cols = list()
    duplicates = dict()
    for col in df.columns:
        dropped = df[col].drop_duplicates()
        if not col.endswith('name') and len(dropped) == 1:
            duplicates[col] = dropped[0]
        else:
            keep_cols.append(col)
    df = df[keep_cols]
    return df, pd.Series(duplicates)


def make_summary(save_dir, result_dir, ops=['mean', 'std', 'max', 'min']):
    """> merge_to_csv(save_dir, result_dir, ops=['mean', 'std', 'max', 'min'])
    Merge all results under 'result_dir' and save summary to 'save_dir'.
    'ops' is a list of operations that will be applied on metrics result.
    """
    df, duplicates = merge_result(result_dir, ops)
    prefix = '%s%s(%s)' % (strftime("[%Y-%m-%d-%H-%M-%S]"),'EXP', os.path.split(result_dir)[1])
    df.to_csv('%sMain.csv' % prefix)
    if len(duplicates) > 0:
        duplicates.to_csv('%sInfo.csv' % prefix)


def _check_dir(save_dir):
    if not os.path.exists(save_dir):
        logger.info("'%s' does not exist. Create it..." % save_dir)
        os.makedirs(save_dir)
