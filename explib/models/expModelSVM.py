from ..base import expModel, Option
from sklearn.svm import SVC
from bunch import Bunch
import logging


logger = logging.getLogger(__name__)


class ModelOption(Option):

    def set_default(self):
        self._name = 'SVM'
        self.C = 1.0
        self.kernel = 'rbf'


class expModelSVM(expModel):

    def __init__(self, **kwargs):
        self._opts = ModelOption(**kwargs)

    def fit(self, data):
        opts = self._opts
        clf = SVC(C=opts.C, kernel=opts.kernel)
        clf.fit(data.train_X, data.train_y)
        pred_y = clf.predict(data.test_X)
        result = Bunch(pred_y=pred_y)
        return result
