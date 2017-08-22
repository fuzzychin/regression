import pandas as pd

import numpy as np

import statsmodels.formula.api as smf

import statsmodels.graphics.api as smg

import seaborn as sns

import matplotlib.pyplot as plt

# Import the data
faa_df = pd.read_csv("faa.csv")

# EDA Lets look at our data
# (What are the variables?)
print(faa_df)

# Future calculations won't work on strings, so let's convert "airbus" and "boeing" to 0 nd 1, respectively
faa_df = faa_df.replace(to_replace="airbus", value=0)
faa_df = faa_df.replace(to_replace="boeing", value=1)

# Check results
print(faa_df)

# Create a correlation matrix
# (Note the results for speed_air)
# np.corrcoef treats each row as a variable, so we use .T to transpose our data_frame
corr_matrix = np.corrcoef(faa_df.T)
smg.plot_corr(corr_matrix, xnames=list(faa_df))
plt.show()

# Remove NaN values and run correlation matrix again
faa_df_no_na = faa_df.dropna()

# Check results
# (How much data is left?)
print(faa_df_no_na)

# Create our correlation matrix again
corr_matrix = np.corrcoef(faa_df_no_na.T)
smg.plot_corr(corr_matrix, xnames=list(faa_df_no_na))
plt.show()

# Lets remove "speed_air" from our original data set
# because correlation between speed_air and speed_ground is high,
# and because removing NaN values would greatly reduce the number of samples
del faa_df["speed_air"]

# Check to make sure column was deleted
print(faa_df)

# Create our correlation matrix with "speed_air" removed
corr_matrix = np.corrcoef(faa_df.T)
smg.plot_corr(corr_matrix, xnames=list(faa_df))
plt.show()

# Pairwise plot using seaborn
sns.pairplot(faa_df)
plt.show()

# Build model with distance as dependant variable and all other variables as independent variables
# (Are any of the P values too high?)
model = smf.ols(formula="distance ~ aircraft + duration + no_pasg + speed_ground + height + pitch", data=faa_df)
results = model.fit()
print(results.summary())

# Build revised model with distance as dependant variable and significant variables as independent variables
revised_model = smf.ols(formula="distance ~ aircraft + speed_ground + height", data=faa_df).fit()
print(revised_model.summary())





