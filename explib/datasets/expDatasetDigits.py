from ..base import expDataset, Option
from sklearn.datasets import load_digits
from bunch import Bunch
import logging


logger = logging.getLogger(__name__)


class DatasetOption(Option):

    def set_default(self):
        self.name = 'Digits'
        self.nb_classes = 10


class expDatasetDigits(expDataset):

    def __init__(self, **kwargs):
        self._opts = DatasetOption(**kwargs)

    def load(self):
        opts = self._opts
        logger.info('Loading Digits...')
        X, y = load_digits(opts.nb_classes, True)
        logger.info('Done')
        data = Bunch(all_X=X, all_y=y)
        return data
