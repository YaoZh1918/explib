from ..base import expModel, Option
from sklearn.linear_model import LogisticRegression
from bunch import Bunch
import logging


logger = logging.getLogger(__name__)


class ModelOption(Option):

    def set_default(self):
        self.name = 'LR'
        self.C = 1.0

class expModelLR(expModel):

    def __init__(self, **kwargs):
        self._opts = ModelOption(**kwargs)

    def fit(self, data):
        opts = self._opts
        clf = LogisticRegression(C=opts.C)
        clf.fit(data.train_X, data.train_y)
        pred_y = clf.predict(data.test_X)
        result = Bunch(pred_y=pred_y)
        return result
