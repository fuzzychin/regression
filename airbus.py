import pandas as pd

import numpy as np

import statsmodels.formula.api as smf

import statsmodels.graphics.api as smg

import seaborn as sns

import matplotlib.pyplot as plt

# Import the data
faa_df = pd.read_csv("faa.csv")

# EDA Lets look at our data
print(faa_df)

# Future calculations won't work on strings, so let's convert "airbus" and "boeing" to 0 nd 1, respectively
faa_df = faa_df.replace(to_replace="airbus", value=0)
faa_df = faa_df.replace(to_replace="boeing", value=1)

# Lets remove "speed_air" from our original data set
del faa_df["speed_air"]

# Create a new data frame for boeing data
airbus_df = faa_df[faa_df.aircraft == 0]

# Examine data
print(airbus_df)

# Create our correlation matrix
corr_matrix = np.corrcoef(airbus_df.T)
smg.plot_corr(corr_matrix, xnames=list(airbus_df))
plt.show()

# Pairwise plot using seaborn
sns.pairplot(airbus_df)
plt.show()

# Build model with distance as dependant variable and all other variables as independent variables
model = smf.ols(formula="distance ~ duration + no_pasg + speed_ground + height + pitch", data=airbus_df)
results = model.fit()
print(results.summary())

# Build revised model with distance as dependant variable and significant variables as independent variables
revised_model = smf.ols(formula="distance ~ speed_ground + height + pitch", data=airbus_df).fit()
print(revised_model.summary())
