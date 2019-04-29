from __future__ import print_function
import re
import string
from operator import itemgetter
import os
import numpy as np
import pandas as pd
path = os.path.join('korean_data.csv')
data1 = pd.read_csv(path)
path2 = os.path.join('korean_data2.csv')
data2 = pd.read_csv(path2)
data1['Actors'] = data2['Actors']
data1.to_csv(os.path.join('korean_data.csv'))
