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
