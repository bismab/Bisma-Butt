## BB

You can use the [editor on GitHub](https://github.com/bismab/for_blog/edit/master/README.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

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

[Link](https://gmail.com) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/bismab/for_blog/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
