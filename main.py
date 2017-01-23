from explib.datasets import *
from explib.metrics import *
from explib.models import *
from explib.settings import *
from explib.base import expEnsemble, expPool, expProfile
from explib.utils import ParamsGrid
import logging
from logging.config import fileConfig


fileConfig('logging.conf', disable_existing_loggers=False)
logger = logging.getLogger()


pool = expPool(4)


if 1:  # use expEnsemble to deal with large experiment
    # declare parameters
    digits_para_grid = ParamsGrid(dict(nb_classes=[5, 10]))
    svm_para_grid = ParamsGrid(dict(C=[.1, 1], kernel=['rbf', 'poly']))
    lr_para_grid = ParamsGrid(dict(C=[.5, 2]))
    # assemble
    ensemble = expEnsemble('result', True)
    ensemble.add_model(expModelSVM, svm_para_grid)
    ensemble.add_model(expModelLR, lr_para_grid)
    ensemble.add_dataset(expDatasetIris)
    ensemble.add_dataset(expDatasetDigits, digits_para_grid)
    ensemble.add_metrics(expMetricAcc(), expMetricAvgF1(average='micro'), expMetricAvgF1(average='macro'))
    ensemble.set_setting(expSettingKFold(n_splits=5))
    # add to pool
    pool.add(ensemble)

if 1:  # a simple experiment is more suitable for expProfile
    model = expModelSVM()
    dataset = expDatasetIris()
    metrics = [expMetricAcc(), expMetricAvgF1(average='micro')]
    settings = expSettingKFold(n_splits=10)
    profile = expProfile(dataset, model, metrics, settings, '.', True)
    # add to pool
    pool.add(profile)

pool.run()


