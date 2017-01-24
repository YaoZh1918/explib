from ..base import expModel
from sklearn.svm import SVC
from bunch import Bunch
import logging


logger = logging.getLogger(__name__)


class expModelSVM(expModel):

    def fit(self, data):
        opts = self._opts
        clf = SVC(C=opts.C, kernel=opts.kernel)
        clf.fit(data.train_X, data.train_y)
        pred_y = clf.predict(data.test_X)
        result = Bunch(pred_y=pred_y)
        return result
