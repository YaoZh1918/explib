from ..base import expSetting
from sklearn.model_selection import KFold
import numpy as np


class expSettingKFold(expSetting):

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
