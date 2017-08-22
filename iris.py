import statsmodels.api as sm

import seaborn as sns

import matplotlib.pyplot as plt

# # Load the iris dataset as a dataframe using seaborn
iris_df = sns.load_dataset("iris")

# EDA - print description of data
print(iris_df)

# # Pairwise plot using seaborn
# sns.pairplot(iris_df, hue="species")
# plt.show()
#
# # Scatterplot using seaborn
# sns.lmplot(x="petal_length", y="petal_width", data=iris_df)
# plt.show()

# Build a linear model and calculate summary statistics
x = iris_df[["petal_length"]]
y = iris_df["petal_width"]

# # Note - model is y given x
model = sm.OLS(y, x)
results = model.fit()

# summary() in statsmodels gives R-like statistical output
print(results.summary())

