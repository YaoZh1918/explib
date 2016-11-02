from abc import abstractmethod, ABCMeta
from bunch import Bunch


class expBase(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self.params = Bunch()


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
    def evaluate(self):
        """Evaluate the result
        to be implemented in subclass
        """
        return


class expSetting(expBase):
    __metaclass__ = ABCMeta

    def __init__(self):
        super(expSetting, self).__init__()

    @abstractmethod
    def run(self):
        """Fit data and evaluate the result
        to be implemented in subclass
        """
        return

class profile(expBase):
    
    def __init__(self, model, dataset, setting, metrics, save_dir):
        pass

    def run(self):
        # todo
        pass

