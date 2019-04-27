from __future__ import print_function
import re
import string
from operator import itemgetter
import os
import numpy as np
import pandas as pd

path = os.path.join('korean_data.csv')
data = pd.read_csv(path)
summaries = data['Summary']
summary_starter = str(summaries[88])[:1]
summary_col = {}
for idx, row in summaries.items():
    if str(row)[:1] == summary_starter:
        row = "nan"
    row = str(row).replace("amp;", "")
    summary_col[idx] = row
data["Summary"] = pd.DataFrame.from_dict(summary_col, orient='index')
data.to_csv(os.path.join('korean_data.csv'))
