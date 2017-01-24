from ..base import expMetric
from sklearn.metrics import accuracy_score


class expMetricAcc(expMetric):

    def evaluate(self, data, result):
        score = accuracy_score(data.test_y, result.pred_y)
        self.values.append(score)
