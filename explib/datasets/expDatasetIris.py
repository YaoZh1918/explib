from ..base import expDataset, Option
from bunch import Bunch
from sklearn.datasets import load_iris
import logging


logger = logging.getLogger(__name__)


class DatasetOption(Option):

    def set_default(self):
        self._name = 'Iris'


class expDatasetIris(expDataset):

    def __init__(self, **kwargs):
        self._opts = DatasetOption(**kwargs)

    def load(self):
        logger.info('Loading Iris...')
        X, y = load_iris(True)
        logger.info('Done')
        data = Bunch(all_X=X, all_y=y)
        return data
