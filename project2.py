import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# read from our clean data set from stage 1
df = pd.read_csv('cleanmovies.csv')

# slice data for desired variables
new = df[['budget', 'runtime', 'score']] 
X = new.values  # input variables
y = df.values[:, 5] #slice dataFrame for target variable

# linear regression model
print('Linear Regression model')
# accumulator variables to find the best test size
top_size = 0
top_r2 = 0

# loop through to find most accurate model based on test size
print('R-squared for each test size')
for j in range(10, 20, 1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(j/100), random_state=42)
    temp_regr = linear_model.LinearRegression().fit(X_train, y_train)
    temp_pred = temp_regr.predict(X_test)
    temp_r2 = metrics.r2_score(y_test, temp_pred)
    print(str(j) + '%: ' + str(temp_r2))
    if temp_r2 > top_r2:
        top_r2 = temp_r2
        top_size = j 
    
# Evaluate the best one 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(top_size/100), random_state=42)
regr = linear_model.LinearRegression().fit(X_train, y_train)
y_pred = regr.predict(X_test)
print('----- Best model -----')
print('For test size ' + str(top_size) + '%')
print('Coefficients:', regr.coef_)
# Root mean squared error
mse = metrics.mean_squared_error(y_test, y_pred)
print('Root mean squared error (RMSE):', sqrt(mse))
# R-squared score: 1 is perfect prediction
print('R-squared score:', metrics.r2_score(y_test, y_pred))

# Let's create one sample and predict the gross using our model
sample = [26000000, 115, 8]     #budget, runtime, score
print('----- Sample case -----')
for column, value in zip(list(new), sample):
    print(column + ': ' + str(value))
sample_pred = regr.predict([sample])
print('Predicted gross: ', int(sample_pred))
print('-----------------------')

# k-NN model

# accumulator variables
top_k = 0
top_r = 0
top_s = 0

print('k-NN model')
print('k value and highest R-squared for each test size')
for j in range(10, 20, 1):
  X1_train, X1_test, y1_train, y1_test = train_test_split(X, y, test_size=(j/100), random_state=42)
  N = len(y1_train)
  k = int(sqrt(N))
  temp_k = 0
  r2 = 0
  for i in range(k-10, k+10):
    temp_neigh = neighbors.KNeighborsRegressor(n_neighbors=i).fit(X1_train, y1_train)
    # Use the model to predict X_test
    temp_knn_pred = temp_neigh.predict(X1_test)
    # Root mean squared error
    temp_r = metrics.r2_score(y1_test, temp_knn_pred)
    if temp_r > r2:
        temp_k = i
        r2 = temp_r
    if  temp_r > top_r:
      top_r = temp_r
      top_k = i
      top_s = j
  print('k=' + str(temp_k) + ', '+  str(j) + '%: ' + str(r2))

# evaluate the best one
print('----- Best model -----')
X1_train, X1_test, y1_train, y1_test = train_test_split(X, y, test_size=(top_s/100), random_state=42)  
neigh = neighbors.KNeighborsRegressor(n_neighbors=top_k).fit(X1_train, y1_train)
# Use the model to predict X_test
knn_pred = neigh.predict(X1_test)
topmse = metrics.mean_squared_error(y1_test, knn_pred)
print('Root mean squared error (RMSE):', sqrt(topmse))
# R-squared score: 1 is perfect prediction
print('R-squared score:', metrics.r2_score(y1_test, knn_pred))
print('for test size ' + str(top_size) +'% and k=' + str(top_k))

# Let's create one sample and predict the gross using our model
sample2 = [26000000, 115, 8]     #budget, runtime, score
print('----- Sample case -----')
for column, value in zip(list(new), sample2):
    print(column + ': ' + str(value))
# using our best model
test = neighbors.KNeighborsRegressor(n_neighbors=top_k).fit(X_train, y_train)
sample_pred2 = test.predict([sample2])
print('Predicted gross:', int(sample_pred2))
print('-----------------------')

# graphs
print('----- Pairwise Data -----')

# narrow down our data
budget = df['budget'].values.reshape(-1,1).astype(float)
runtime = df['runtime'].values.reshape(-1,1).astype(float)
score = df['score'].values.reshape(-1,1).astype(float)
gross = df.values[:, 5].astype(float) 


# looking at budget and gross
plt.scatter(budget, gross,  color='gray')

# find linear trendline
budgetmodel = LinearRegression()
budgetmodel.fit(budget, gross)
y_predicted = budgetmodel.predict(budget)

# label plot and add trendline
plt.title("Gross vs Budget")
plt.xlabel('Budget ($)')
plt.ylabel('Gross ($)')
plt.plot(budget, y_predicted)
plt.show()

# model evaluation
r2 = r2_score(gross, y_predicted)
print('Gross vs Budget R-Squared:', r2)

# looking at runtime and gross 
plt.scatter(runtime, gross,  color='gray')

# find linear trendline
runtime_model = LinearRegression()
runtime_model.fit(runtime, gross)
run_predicted = runtime_model.predict(runtime)

# label plot and add trendline
plt.title("Gross vs Runtime")
plt.xlabel('Runtime (minutes)')
plt.ylabel('Gross ($)')
plt.plot(runtime, run_predicted)
plt.show()

# model evaluation
run_r2 = r2_score(gross, run_predicted)
print('Gross vs Runtime R-squared:', run_r2)

# looking at score and gross
plt.scatter(score, gross,  color='gray')

# find linear trendline
score_model = LinearRegression()
score_model.fit(score, gross)
score_predicted = score_model.predict(score)

# label plot and add trendline
plt.plot(score, score_predicted)
plt.title("Gross vs Score")
plt.xlabel('Score')
plt.ylabel('Gross ($)')
plt.show()

# model evaluation
score_r2 = r2_score(gross, score_predicted)
print('Gross vs Score R-squared:', score_r2)


# looking at score and gross the other way
plt.scatter(gross, score,  color='gray')

plt.title("Score vs Gross")
plt.ylabel('Score')
plt.xlabel('Gross ($)')
plt.show()

# model evaluation
score_r2 = r2_score(gross, score_predicted)
print('Gross vs Score R-squared:', score_r2)

# remove movie with highest gross 
rem = df.set_index("name")
rem = rem.drop("Star Wars: The Force Awakens", axis=0)
starwars = rem['budget'].values.reshape(-1,1).astype(float)
y = rem.values[:, 5].astype(float)

# looking at budget and gross minus outlier
plt.scatter(starwars, y,  color='gray')

# find linear trendline
star_model = LinearRegression()
star_model.fit(starwars, y)
star_predicted = star_model.predict(starwars)

# label plot and add trendline
plt.plot(starwars, star_predicted)
plt.title("Gross vs Budget (without outlier)")
plt.xlabel('Budget ($)')
plt.ylabel('Gross ($)')
plt.show()

# model evaluation
star_r2 = r2_score(y, star_predicted)
print('Gross vs Budget (no outlier) R-squared:', star_r2)

# make look at runtime and score based on some ratings
g = df[df['rating'] == 'G']
pg13 = df[df['rating'] == 'PG-13']
R = df[df['rating'] == 'R']
unrate = df[df['rating'] == 'UNRATED']
g_run = g.values[:,9]
g_score = g.values[:,10]
r_run = R.values[:,9]
r_score = R.values[:,10]
p_run = pg13.values[:,9]
p_score = pg13.values[:,10]

# make the graph
plt.scatter(g_score, g_run, color='blue', marker='o', label='G')
plt.scatter(p_score, p_run, color='red', marker='>', label='PG13')
plt.scatter(r_score, r_run, color='green', marker='x', label='R')
plt.title('Score and Runtime of G, PG13, and R rated movies')
plt.xlabel('Score')
plt.ylabel('Runtime')
plt.legend(loc='lower right')
plt.show()


