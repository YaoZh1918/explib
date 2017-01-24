from ..base import expSetting, Option
from sklearn.model_selection import KFold
import numpy as np


class SettingOption(Option):

    def set_default(self):
        self.name = 'KFold'
        self.n_splits = 10


class expSettingKFold(expSetting):

    def __init__(self, **kwargs):
        super(expSettingKFold, self).__init__()
        self._opts = SettingOption(**kwargs)

    def run(self):
        opts = self._opts
        data = self.dataset.load()
        n = data.all_X.shape[0]
        kf = KFold(n_splits=opts.n_splits, shuffle=True)
        for train_idx, test_idx in kf.split(range(n)):
            data.train_X = data.all_X[train_idx, :]
            data.train_y = data.all_y[train_idx]
            data.test_X = data.all_X[test_idx, :]
            data.test_y = data.all_y[test_idx]
            result = self.model.fit(data)
            for metric in self.metrics:
                metric.evaluate(data, result)
