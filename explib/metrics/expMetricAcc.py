from ..base import expMetric, Option
from sklearn.metrics import accuracy_score


class MetricOption(Option):

    def set_default(self):
        self.name = 'Acc'

class expMetricAcc(expMetric):

    def __init__(self):
        super(expMetricAcc, self).__init__()
        self._opts = MetricOption()

    def evaluate(self, data, result):
        score = accuracy_score(data.test_y, result.pred_y)
        self.values.append(score)
