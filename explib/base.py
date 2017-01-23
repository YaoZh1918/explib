from abc import abstractmethod, ABCMeta
import hashlib
import os
import logging
from itertools import chain, product, imap
from multiprocessing import Pool
from .utils import savepkl, ParamsGrid


logger = logging.getLogger(__name__)


class Option(object):

    def __init__(self, **kwargs):
        self.set_default()
        self.update(**kwargs)

    def set_default(self):
        self.name = None

    def __str__(self):
        sorted_pairs = sorted(self.__dict__.iteritems())
        kv_str = ', '.join(map(lambda x: '%s=%s' % x, sorted_pairs))
        return '%s(%s)' % (self.__class__.__name__, kv_str)

    __repr__ = __str__

    def update(self, **kwargs):
        for k, v in kwargs.iteritems():
            if k in self.__dict__:
                self.__dict__[k] = v
            else:
                raise KeyError("'%s' is not a valid parameter." % k)


class expBase(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self._opts = Option()


class expDataset(expBase):
    __metaclass__ = ABCMeta

    def __init__(self):
        super(expDataset, self).__init__()

    @abstractmethod
    def load(self):
        """Load data
        to be implemented in subclass
        """
        return


class expModel(expBase):
    __metaclass__ = ABCMeta

    def __init__(self):
        super(expModel, self).__init__()

    @abstractmethod
    def fit(self, data):
        """Fit model to the data
        to be implemented in subclass
        """
        return


class expMetric(expBase):
    __metaclass__ = ABCMeta

    def __init__(self):
        super(expMetric, self).__init__()
        self.values = []

    @abstractmethod
    def evaluate(self, dataset, model):
        """Evaluate the result
        to be implemented in subclass
        """
        return


class expSetting(expBase):
    __metaclass__ = ABCMeta

    def __init__(self):
        super(expSetting, self).__init__()
        self.dataset = None
        self.model = None
        self.metrics = []

    def setup(self, dataset, model, metrics):
        self.dataset = dataset
        self.model = model
        self.metrics = metrics

    @abstractmethod
    def run(self):
        """Fit data and evaluate the result
        to be implemented in subclass
        """
        return

    def get_metrics_result(self):
        result = list()
        for metric in self.metrics:
            result.append((metric._opts, metric.values))
        return result


class expProfile(expBase):

    def __init__(self, dataset, model, metrics, setting,
                 save_dir, overwrite=False):
        self.dataset = dataset
        self.model = model
        self.metrics = metrics
        self.setting = setting
        self.save_dir = save_dir
        self.overwrite = overwrite

    def run(self):
        # generate file name
        opts_list = map(lambda x: x._opts,
                        [self.dataset, self.model,
                         self.setting] + self.metrics)
        encoder = hashlib.md5()
        encoder.update(';'.join(map(str, opts_list)))
        filename = os.path.join(self.save_dir, encoder.hexdigest())
        # check existence
        if not self.overwrite and os.path.exists(filename):
            logger.warn("'%s' already exists, skip." % filename)
            return
        # run and save
        self.setting.setup(self.dataset, self.model, self.metrics)
        setting_result = self.setting.run()
        metrics_result = self.setting.get_metrics_result()
        result = dict(Options=opts_list, Metrics=metrics_result,
                      Setting=setting_result)
        try:
            savepkl(result, filename)
            logger.info("'%s' saved." % filename)
        except IOError:
            logger.error("IOError when saving '%s'" % filename)


class expEnsemble(expBase):

    def __init__(self, save_dir, overwrite=False):
        self.models = []
        self.datasets = []
        self.metrics = []
        self.setting = None
        self.save_dir = save_dir
        self.overwrite = overwrite
        self._n_models = 0
        self._n_datasets = 0

    def __len__(self):
        return self._n_models * self._n_datasets

    def add_model(self, model, para_grid=ParamsGrid()):
        self.models.append(imap(lambda para: model(**para), para_grid))
        self._n_models += max(1, len(para_grid))

    def add_dataset(self, dataset, para_grid=ParamsGrid()):
        self.datasets.append(imap(lambda para: dataset(**para), para_grid))
        self._n_datasets += max(1, len(para_grid))

    def add_metrics(self, *args):
        self.metrics.extend(args)

    def set_setting(self, setting):
        self.setting = setting

    def __iter__(self):
        models = chain(*self.models)
        datasets = chain(*self.datasets)
        for model, dataset in product(models, datasets):
            profile = expProfile(dataset, model, self.metrics,
                                 self.setting, self.save_dir, self.overwrite)
            yield profile


class expPool(expBase):

    def __init__(self, n_workers=2):
        self.n_workers = n_workers
        self.tasks = list()
        self.dirs = set()

    def add(self, profiles):
        self.dirs.add(profiles.save_dir)
        if isinstance(profiles, expProfile):  # single profile
            self.tasks.append([profiles])
        else:
            self.tasks.append(profiles)

    def __len__(self):
        return sum(map(len, self.tasks))

    def run(self):
        map(_check_dir, self.dirs)
        logger.info('# Experiments: %3d' % len(self))
        pool = Pool(self.n_workers)
        pool.map(_wrapper, enumerate(chain(*self.tasks)))


def _wrapper(args):
    i, foo = args
    logger.info('Exp %3d Begins...' % i)
    try:
        foo.run()
    except BaseException, e:
        logger.error('Exp %3d: %s' % (i, e))
    logger.info('Exp %3d Done!' % i)


def _check_dir(save_dir):
    if not os.path.exists(save_dir):
        logger.info("'%s' does not exist. Create it..." % save_dir)
        os.makedirs(save_dir)
