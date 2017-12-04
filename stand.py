from __future__ import print_function

import csv
import numpy as np

from model import get_model
from utils import real_to_cdf, preprocess

X = preprocess(X)
pred_systole = model_systole.predict(X, batch_size=batch_size, verbose=1)


if ef > 80:
    print("High EF")

# https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
