from abc import abstractmethod, ABCMeta
from bunch import Bunch
import hashlib
import os
import logging
from .utils import savepkl


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
            result.append((metric._opts,metric.values))
        return result


class expProfile(expBase):

    def __init__(self, dataset, model, metrics, setting, save_dir):
        self.dataset = dataset
        self.model = model
        self.metrics = metrics
        self.setting = setting
        self.save_dir = save_dir

    def run(self, overwrite=False):
        # generate file name
        opts_list = map(lambda x: x._opts,
                   [self.dataset, self.model,
                    self.setting] + self.metrics)
        encoder = hashlib.md5()
        encoder.update(';'.join(map(str, opts_list)))
        filename = os.path.join(self.save_dir, encoder.hexdigest())
        # check existence
        if not overwrite and os.path.exists(filename):
            logger.warn('%s already exists, skip.' % filename)
            return
        # run and save
        self.setting.setup(self.dataset, self.model, self.metrics)
        setting_result = self.setting.run()
        metrics_result = self.setting.get_metrics_result()
        result = dict(Options=opts_list, Metrics=metrics_result,
                      Setting=setting_result)
        try:
            savepkl(result, filename)
            logger.info('%s saved.' % filename)
        except IOError:
            logger.error('IOError when saving %s' % filename)
