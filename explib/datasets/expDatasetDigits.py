from ..base import expDataset
from sklearn.datasets import load_digits
from bunch import Bunch
import logging


logger = logging.getLogger(__name__)


class expDatasetDigits(expDataset):

    def load(self):
        opts = self._opts
        logger.info('Loading Digits...')
        X, y = load_digits(opts.nb_classes, True)
        logger.info('Done')
        data = Bunch(all_X=X, all_y=y)
        return data
