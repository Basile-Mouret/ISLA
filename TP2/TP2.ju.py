# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: isla2025
#     language: python
#     name: isla2025
# ---


# %% [markdown]
# ### ENSIMAG – Grenoble INP – UGA - Academic year 2025-2026
# # Introduction to Statistical Learning and Applications ([website](https://github.com/ISLA-Grenoble/2025-main))
#
# - Pedro L. C. Rodrigues -- `pedro.rodrigues@inria.fr`
#
# - Isabella Costa Maia -- `isabella.costa-maia@grenoble-inp.fr`
#
# - Pierre Marrec -- `pierre.marrec@inria.fr`
#
# ***
#
# ### ⚠️ General guidelines for TPs
#
# The report should contain graphical representations and explanatory text. For each graph, axis names should be provided as well
# as a legend when it is appropriate. Figures should be explained by a few sentences in the text. Answer to
# the questions in order and refer to the question number in your report. Computations and
# graphics have to be performed in `python`. The report should be written as a jupyter notebook. This is a file format that allows users to format documents containing text written in markdown and `python` instructions. You should include all of the `python` instructions that you have used in the document so that it may be possible to replicate your results.
#
# ***
#
# # 🖥️ TP2: Principal components regression in genetics
#
# The goal of this TP session is to use genetic markers to predict the geographical origin of a set of indians from South, Central, and North America. We propose to build two regression linear models to predict the latitude and longitude of an individual based on its genetic markers. Because the number of markers (p = 5709) is larger than the number of samples (N = 494), the predictors of the regression model will be the outputs of a principal component analysis (PCA) performed on the genetic markers. A genetic marker is encoded 1 if the individual has a mutation, 0 elsewhere.
#
# ## ▶️ Exercise 1: Data visualization (1 point)
#
# NB: To do this exercise you will have to install packages `geopandas` and `geodatasets`.
#
# Download dataset `NAm2.txt` from [here](https://github.com/ISLA-Grenoble/2025-main/blob/main/TP/TP2/NAm2.txt). Each row of the dataset corresponds to an individual and the columns have explicit names. The third column contains the names of the tribes to which each individual pertains. Columns 7 and 8 contain the latitude and the longitude and from Column 9 onwards are genetic markers, which are encoded are 0 or 1. Run the code described below and explain how it works.
#
# ```
# import pandas as pd
# import geopandas as gpd
# import geodatasets
# import matplotlib.pyplot as plt
#
# # Load the data
# file_path = 'NAm2.txt'
# df = pd.read_csv(file_path, delimiter=' ')
#
# # Extract relevant columns
# latitude = df.iloc[:, 6]
# longitude = df.iloc[:, 7]
# tribes = df.iloc[:, 2]
#
# # Create a GeoDataFrame
# gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(longitude, latitude))
#
# # Plotting
# world = gpd.read_file(geodatasets.get_path('naturalearth.land'))
# fig, ax = plt.subplots(figsize=(8.0, 6.5))
# plt.subplots_adjust(left=0.0, right=0.90, bottom=0.10, top=0.92)
# world.clip([-140, -55, -25, 75]).plot(ax=ax, color='white', edgecolor='black')
# marker_list = ['o', 'v', 's']
# colors_list = [f'C{i}' for i in range(9)]
# for i, tribe in enumerate(gdf['Pop'].unique()):
#     members_tribe = gdf[gdf['Pop'] == tribe]
#     ax.scatter(members_tribe['long'], members_tribe['lat'], 
#                marker=marker_list[i//9], 
#                color=colors_list[i%9], label=tribe)
# ax.legend(loc='center right', bbox_to_anchor=(1.4, 0.5))
# ax.set_title('Tribes Locations')
# ax.set_xlabel('Longitude')
# ax.set_ylabel('Latitude')
# fig.show()
# ```

# %%

# %% [markdown]
# ## ▶️ Exercise 2: Multiple linear regression (2 points)
#
# Using **only** the genetic markers as predictors, you will estimate a multiple linear regression model to predict the longitude of each individual.
#
# You will proceed in several steps.
#
# **(a)** First, try to estimate the coefficients of the multiple linear regression using the expression seen in class 
#
# $$\\hat{\\beta} = (X^\\top X)^{-1}X^\\top y$$
#
# You should proceed as we did in TP1 using `numpy.linalg.solve` to obtain the values of $\\beta$. 
#
# Did you run into any errors? What is going on? Relate your answer to the fact that $\\text{rank}(X) < p$, where $X \\in R^{N*p}$ is the data matrix.

# %%

# %% [markdown]
# **(b)** Use function `numpy.linalg.lstsq` to estimate the coefficients (it may take a few seconds to get a result). 
#
# And now? Did you get any errors? Why is that? 
#
# Relate your answer to the difference between functions `numpy.linalg.solve` and `numpy.linalg.lstsq`.
#
# You can check the documention for both functions as well as [this](https://netlib.org/lapack/lug/node27.html) link for more information.

# %%

# %% [markdown]
#  **(c)** We will now use `sklearn` to do our linear regression with the help of class `sklearn.linear_model.LinearRegression` whose documentation is available [here](https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LinearRegression.html). Note that every estimator from `sklearn` has a `fit` and a `predict` method, which are used to calculate coefficients and predict values (see [here](https://scikit-learn.org/stable/getting_started.html#fitting-and-predicting-estimator-basics) for more info). In our current case, we can do:
#
# ```
# # select only the genetic markers as predictors
# predictors = df.columns[8:]
# # create the design matrix
# X = df[predictors].values
# # get the observed values to predict
# y = df['long']
# # fit a multiple linear regression model
# lr = LinearRegression()
# lr.fit(X, y)
# ```
#
# You should not run into errors now, since `sklearn` also uses `lstsq` to solve the normal equations, as shown [here](https://github.com/scikit-learn/scikit-learn/blob/d666202a9349893c1bd106cc9ee0ff0a807c7cf3/sklearn/linear_model/_base.py#L682) (though it uses the `scipy` implementation instead of the `numpy` for "historical" reasons). Check the values of the estimated coefficients stored as an attribute in `lr.coef_`, are they the same as the ones obtained in item **(b)**? Probably not. This is because `sklearn` re-centers the predictors before estimating the coefficients of the linear regression, as shown [here](https://github.com/scikit-learn/scikit-learn/blob/d666202a9349893c1bd106cc9ee0ff0a807c7cf3/sklearn/linear_model/_base.py#L622). What would be a practical reason for doing such re-centering systematically? Hint: it has to do with how to interpret the intercept of the model. 

# %%

# %% [markdown]
# ## ▶️ Exercise 3: Principal components analysis (5 points)
#
# **(a)** Explain in a few words the main concepts and ideas underlying the principal component analysis (PCA). You should include both the geometric and statistical interpretations of PCA.

# %%
print("hello world")
# %% [markdown]
# **(b)** Use the estimator defined in `sklearn.decomposition.PCA` to do a PCA on the dataset. Plot the first two dimensions of the projected data points on a scatterplot. The scattered points should have different markers and colors depending on which tribe they belong to. You can use the same color/marker style from **Exercise 2** or propose a new one.

# %%

# %% [markdown]
# **(c)** Remember from our class that the results of PCA are affected when pre-processing transformations are applied to the data. We will illustrate this using `sklearn.preprocessing.StandardScaler` as per:
# ```
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# scaler.fit(X)
# X_std = scaler.transform(X)
# ```
# Redo the 2D scatter plot from item **(b)** on the normalized version of the datast. How does it compare to your previous plot?

# %%

# %% [markdown]
# **(d)** Given the results in **(b)** and **(c)**, what can you conclude regarding the necessity of standardizing the data points for the dataset consider in this TP?

# %%

# %% [markdown]
# **(e)** Which percentage of variance is captured by the first two principal components? How many principal components would you keep if you would like to represent the genetic markers using a minimal number of principal components? To help answering this question, you can use a plot showing the cumulative percentage of variance as a function of the number of principal components.

# %%

# %% [markdown]
# ## ▶️ Exercise 4: Principal components regression (4 points)
#
# **(a)** Predict the latitude and the longitude of all points from the dataset using the scores of the first 250 PCA axes. Plot the predicted spatial coordinates using the same style and structure from **Exercise 1** and compare the results from each plot. What can you conclude? Does the new map illustrate somehow too optimistically (or too pessimistically) the ability to find geographical origin of individuals outside the database from its genetic markers? Justify your answer.

# %%

# %% [markdown]
# **(b)** Quantify the error of the linear regression model using the mean distance between real and predicted coordinates. Beware to use `sklearn.metrics.pairwise.haversine_distances` so to correctly measure the distances between points so to take into account the curvature of the Earth. Your answer should be given in kilometers.

# %%

# %% [markdown]
# ## ▶️ Exercise 5: PCR and cross-validation (6 points)
#
# Our goal now is to build the best model to predict individual geographical coordinates. 
#
# For this, you will run a linear regression to predict latitudes and longitudes. Note that `sklearn.linear_model.LinearRegression` can naturally handle the fact of having two sets of coefficients. We will use ten-fold cross-validation to helps us choose the number of principal axes that we should keep. You should report the errors in terms of kilometers as done in **Exercise 4(b)**.
#
# **(a)** Recall in a few words the principle of cross-validation. Explain why this procedure is useful when building a predictive model. Your answer should mention different strategies to handle datasets in which the samples are not IID.

# %%

# %% [markdown]
# **(b)** Based on the structure of the dataset being used, such as the different countries of the individuals and the order in which the rows of the dataframe are provided, explain which choice of cross-validation iterator from [here](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators) seems the most adequate for our context.

# %%

# %% [markdown]
# **(c)** We first assess the quality of the PCR fit for `n_components=4`. Note that you should be careful in avoiding [data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage) problems when doing the PCA followed by a multiple linear regression. You should use the pipeline interface from scikit-learn with `sklearn.pipeline.make_pipeline` to facilitate your task. Be sure to evaluate the errors as done in **Exercise 4(b)**.

# %%
# %% [markdown]
# **(d)** Repeat the analysis from item **(b)** but changing `n_components` between 2 and 440 in steps of 10. Plot the mean training and test errors versus the number of principal components. Attention, the errors should be given in kilometers.

# %%

# %% [markdown]
# **(e)** Which model would you keep? What is the prediction error for this model? Compare it with its corresponding training error. Plot the predicted coordinates on a map as in **Exercise 4(a)**. What can you conclude?

# %%

# %% [markdown]
# ## ▶️ Exercise 6: Conclusion (2 points)
#
# Propose a conclusion to your study. You can write a paragraph about the quality of predictors versus the number of factors, possible improvements to the approach (for instance, showing what happens when using [partial least squares](https://scikit-learn.org/1.5/auto_examples/cross_decomposition/plot_pcr_vs_pls.html) instead of PCR), comment on the performance of the regression in predictions for each country separately, etc. Note that we expect a thorough presentation of the final predictive model as well as an interpretation of it, not simply a bunch of `python` code lines.

# %%
