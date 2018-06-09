# Bisma's Projects 

## Python 
[Exploring Lending Club Data & Predicting Scores via Logistic Regression](https://github.com/bismab/LendingClub_Expl_LogReg/blob/master/Term%20Project%20v.13%20(Tuning%20with%20%26%20without%20RFE)%20Grid%20Search.ipynb)

## R


## Project 1

The data analysis study will explore the temperature increase trends and compare them to emission patterns for the five most populated states in the USA; California, Texas, New York, Illinois and Florida.

Two separate time-series datasets were used for the purpose of the project. The first dataset, “GlobalLandTemperaturesByState.csv” or DS1 (Appendix A), consisted of average monthly temperatures dated from 1855 to 2013 from states across the globe. The second dataset, “CO2 Emissions by State.csv” or DS2 (Appendix A), consisted of total emissions by type, average carbon footprint per person and total population, dated from 1960 to 2001.

### Importing libraries & functions

```markdown
import pandas as pd
import statsmodels.api as sm
import numpy as np
pd.options.display.max_rows = 15
import datetime as dt
from __future__ import division
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy.random as rndn
from __future__ import division
from numpy.random import randn
import os
np.random.seed(12345)
plt.rc('figure', figsize=(10, 6))
np.set_printoptions(precision=4)
import matplotlib
matplotlib.style.use('ggplot')

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

```
```markdown
dfStateTemp=pd.read_csv("GlobalLandTemperaturesByState.csv")
dfStateTemp
```
