# MATHS
import pandas as pd
import numpy as np
import seaborn as sns
from numpy.fft import fft
from scipy import stats
from scipy.fft import fft2, ifft2

# Plotting
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.optimize import curve_fit
from IPython.display import HTML
import altair as alt

# OS
from time import time
import os
from pathlib import Path

# Own libraries
from nfc_signal_offline import *
from nfc_signal_help import *
from device import *

np.random.seed(52102) # always use the same random seed to make results comparable
