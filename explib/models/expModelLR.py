from ..base import expModel
from sklearn.linear_model import LogisticRegression
from bunch import Bunch
import logging


logger = logging.getLogger(__name__)


class expModelLR(expModel):

    def fit(self, data):
        opts = self._opts
        clf = LogisticRegression(C=opts.C)
        clf.fit(data.train_X, data.train_y)
        pred_y = clf.predict(data.test_X)
        result = Bunch(pred_y=pred_y)
        return result
