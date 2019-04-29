from __future__ import print_function
import re
import string
from operator import itemgetter
import os
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import math
import json
from nltk.stem import PorterStemmer

path = os.path.join(os.getcwd(),"app", "irsystem", "models",'korean_data.csv')
korean_data = pd.read_csv(path)
new_korean_data = korean_data
for idx, value in korean_data.iterrows():
    summary = value["Summary"]
    
