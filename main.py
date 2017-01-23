from explib.datasets import *
from explib.metrics import *
from explib.models import *
from explib.settings import *
from explib.base import expProfile
import logging
from logging.config import fileConfig


fileConfig('logging.conf', disable_existing_loggers=False)
logger = logging.getLogger()


def exp_Iris():
    dataset = expDatasetIris()
    model = expModelSVM()
    metrics = [expMetricAcc(), expMetricAvgF1(average='micro'), expMetricAvgF1(average='macro')]
    setting = expSettingKFold(n_splits=5)
    profile = expProfile(dataset, model, metrics, setting, '.')
    profile.run()


if 1:
    exp_Iris()
