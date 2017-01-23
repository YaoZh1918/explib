from ..base import expMetric, Option
from sklearn.metrics import f1_score


class MetricOption(Option):

    def set_default(self):
        self.name = 'F1'
        self.average = 'micro'


class expMetricAvgF1(expMetric):

    def __init__(self, **kwargs):
        super(expMetricAvgF1, self).__init__()
        self._opts = MetricOption(**kwargs)

    def evaluate(self, data, result):
        opts = self._opts
        score = f1_score(data.test_y, result.pred_y, average=opts.average)
        self.values.append(score)
