from ..base import expMetric
from sklearn.metrics import f1_score


class expMetricAvgF1(expMetric):

    def evaluate(self, data, result):
        opts = self._opts
        score = f1_score(data.test_y, result.pred_y, average=opts.average)
        self.values.append(score)
