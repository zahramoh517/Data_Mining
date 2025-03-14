import pandas as pd  
import numpy as np  
from sklearn.datasets import fetch_california_housing  
from sklearn.model_selection import train_test_split, GridSearchCV  
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LogisticRegression  
from sklearn.preprocessing import StandardScaler  
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc,RocCurveDisplay 
from sklearn import linear_model
import matplotlib.pyplot as plt  
import seaborn as sns  
from sklearn.decomposition import PCA  
from sklearn.cluster import KMeans  
from sklearn.mixture import GaussianMixture  
from sklearn.metrics import silhouette_score 



data = fetch_california_housing(as_frame=True) #loading the data as a datafram

data_df = data.data  
target = data.target #getting the target 'MedHouseVal'
data_df['MedHouseVal'] = target 
#print(data_df.head())

#Explore the Datase
print(data_df.shape)
print(data_df.columns)
print(data_df.dtypes)
print(data_df.describe())

#isualize all feature
pdhist = data_df.hist()

#Handle Missing Data
num_missing = int(len(data_df) * 0.1)  #getting the val of 10%
random_indices = np.random.choice(data_df.index, size=num_missing, replace=False) #selcting random rows
data_df.loc[random_indices,['AveRooms' ,'AveOccup']] = np.nan #ntroduce missing values in 10% of the rows

#Replace missing data with the mean of the feature values
mean_room = data_df['AveRooms'].mean()
mean_occup = data_df['AveOccup'].mean()

data_df['AveRooms'] = data_df['AveRooms'].fillna(mean_room)
data_df['AveOccup'] = data_df['AveOccup'].fillna(mean_occup)

#Feature Scaling
scaler = StandardScaler()
features = data_df.drop(columns=['MedHouseVal']) #droping the target so it stays the same
target = data_df['MedHouseVal'] #storing the target 
#print(target.head()) 

scaled_features = scaler.fit_transform(features)
df_scaled = pd.DataFrame(scaled_features, columns=features.columns) #turning the scaled data back to df
df_scaled['MedHouseVal'] = data_df['MedHouseVal'] #adding the target col back 
df_scaled.head()

#Assaining the X and y values  
X = df_scaled.drop(columns=['MedHouseVal'])
y = df_scaled['MedHouseVal']

#Divide the dataset into training (80%) and testing (20%) subsets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_lin = linear_model.predict(X_test)

#Train Lasso Regression
lasso = Lasso()
lasso.fit(X_train, y_train)


#Useing grid search (’alpha’: [0.01, 0.1, 1, 10]) and 5-fold cross-validation to tune hyperparameters
param_grid = {'alpha': [0.01, 0.1, 1, 10]}
lasso_grid_search = GridSearchCV(lasso, param_grid, cv=5, n_jobs=-1)
lasso_grid_search.fit(X_train, y_train)

#Finding the best alpha
best_lasso = lasso_grid_search.best_estimator_
y_pred_lasso = best_lasso.predict(X_test)

# Evaluate Linear Regression
mse_lin = mean_squared_error(y_test, y_pred_lin)
mae_lin = mean_absolute_error(y_test, y_pred_lin)
r2_lin = r2_score(y_test, y_pred_lin)

# Evaluate Lasso Regression
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
mae_lasso = mean_absolute_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

# Evaluate Ridge Regression
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

print(f"Linear Regression: MSE={mse_lin}, MAE={mae_lin}, R^2={r2_lin}")
print(f"Lasso Regression: MSE={mse_lasso}, MAE={mae_lasso}, R^2={r2_lasso}")
print(f"Ridge Regression: MSE={mse_ridge}, MAE={mae_ridge}, R^2={r2_ridge}")

#Reformating the data 
y_binary = (y > y.median()).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

#Perform Logistic Regression
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)
y_pred = logistic_model.predict(X_test)

#Evaluate
print("\nLogistic Regression:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, logistic_model.predict_proba(X_test)[:, 1]))