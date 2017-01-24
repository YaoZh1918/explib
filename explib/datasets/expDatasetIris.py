from ..base import expDataset
from bunch import Bunch
from sklearn.datasets import load_iris
import logging


logger = logging.getLogger(__name__)


class expDatasetIris(expDataset):

    def load(self):
        logger.info('Loading Iris...')
        X, y = load_iris(True)
        logger.info('Done')
        data = Bunch(all_X=X, all_y=y)
        return data
