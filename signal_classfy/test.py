import numpy as np

from signal_classfy.data_init import data_test_init
from signal_classfy.typing.Classify_enum import classes


def test(model):
    QPSK = data_test_init()
    test_predict = model.predict(QPSK)
    print(test_predict)
