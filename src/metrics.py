import numpy as np

from pytorch_tabnet.metrics import Metric
from scipy.special import expit


class LogitsLogLoss(Metric):

    def __init__(self):
        self._name = 'logits_ll'
        self._maximize = False

    def __call__(self, y_true, y_pred):
        aux = (1 - y_true) * np.log(1 - expit(y_pred) + 1e-15) + y_true * np.log(expit(y_pred) + 1e-15)
        return np.mean(-aux)