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
import plotly.plotly as py
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.basemap import Basemap
import numpy as np
%matplotlib inline  
import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder
```

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

**Metadata**
- annual_inc: The self-reported annual income provided by the borrower during registration.
- annual_inc_joint: The combined self-reported annual income provided by the co-borrowers during registration
- application_type: Indicates whether the loan is an individual application or a joint application with two co-borrowers
- collections_12_mths_ex_med: Number of collections in 12 months excluding medical collections
- delinq_2yrs: The number of 30+ days past-due incidences of delinquency in the borrower's credit file for the past 2 years
- desc: Loan description provided by the borrower
- dti: A ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower’s self-reported monthly income.
- dti_joint: A ratio calculated using the co-borrowers' total monthly payments on the total debt obligations, excluding mortgages and the requested LC loan, divided by the co-borrowers' combined self-reported monthly income
- emp_length: Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years.
- emp_title: The job title supplied by the Borrower when applying for the loan.*
- funded_amnt: The total amount committed to that loan at that point in time.
- funded_amnt_inv: The total amount committed by investors for that loan at that point in time.
- grade: Lending Club assigned loan grade
- home_ownership: The home ownership status provided by the borrower during registration. Our values are: RENT, OWN, MORTGAGE, OTHER.
- initial_list_status: The initial listing status of the loan. Possible values are – W, F
- inq_last_6mths: The number of inquiries in past 6 months (excluding auto and mortgage inquiries)
- installment: The monthly payment owed by the borrower if the loan originates.
- int_rate: Interest Rate on the loan
- is_inc_v: Indicates if income was verified by LC, not verified, or if the income source was verified
- last_credit_pull_d: The most recent month LC pulled credit for this loan
- last_pymnt_amnt: Last total payment amount received
- loan_status: Current status of the loan
- mths_since_last_delinq: The number of months since the borrower's last delinquency.
- mths_since_last_major_derog: Months since most recent 90-day or worse rating
- mths_since_last_record: The number of months since the last public record.
- next_pymnt_d: Next scheduled payment date
- open_acc: The number of open credit lines in the borrower's credit file.
- out_prncp: Remaining outstanding principal for total amount funded
- out_prncp_inv: Remaining outstanding principal for portion of total amount funded by investors
- pub_rec: Number of derogatory public records purpose A category provided by the borrower for the loan request.
- pymnt_plan: Indicates if a payment plan has been put in place for the loan
- recoveries: post charge off gross recovery
- revol_bal: Total credit revolving balance
- revol_util: Revolving line utilization rate, or the amount of credit the borrower is using relative to all available revolving credit.
- term: The number of payments on the loan. Values are in months and can be either 36 or 60.
- total_acc: The total number of credit lines currently in the borrower's credit file
- total_pymnt: Payments received to date for total amount funded
- total_pymnt_inv: Payments received to date for portion of total amount funded by investors
- total_rec_int: Interest received to date
- total_rec_late_fee: Late fees received to date
- total_rec_prncp: Principal received to date
- verified_status_joint: Indicates if the co-borrowers' joint income was verified by LC, not verified, or if the income source was verified
- open_acc_6m: Number of open trades in last 6 months
- open_il_6m: Number of currently active installment trades
- open_il_12m: Number of installment accounts opened in past 12 months
- open_il_24m: Number of installment accounts opened in past 24 months
- mths_since_rcnt_il: Months since most recent installment accounts opened
- total_bal_il: Total current balance of all installment accounts
- il_util: Ratio of total current balance to high credit/credit limit on all install acct
- open_rv_12m: Number of revolving trades opened in past 12 months
- open_rv_24m: Number of revolving trades opened in past 24 months
- max_bal_bc: Maximum current balance owed on all revolving accounts
- all_util: Balance to credit limit on all trades
- total_rev_hi_lim:   Total revolving high credit/credit limit
- inq_fi: Number of personal finance inquiries
- total_cu_tl: Number of finance trades
- inq_last_12m: Number of credit inquiries in past 12 months
- acc_now_delinq: The number of accounts on which the borrower is now delinquent.
- tot_coll_amt: Total collection amounts ever owed
- tot_cur_bal: Total current balance of all accounts
