# subasri
#earthquake prediction model using python
Index(['Date', 'Time', 'Latitude', 'Longitude', 'Type', 'Depth', 'Depth Error', 'Depth Seismic Stations', 'Magnitude', 'Magnitude Type', 'Magnitude Error', 'Magnitude Seismic Stations', 'Azimuthal Gap', 'Horizontal Distance', 'Horizontal Error', 'Root Mean Square', 'ID', 'Source', 'Location Source', 'Magnitude Source', 'Status'], dtype='object')
Figure out the main features from earthquake data and create a object of that features, namely, Date, Time, Latitude, Longitude, Depth, Magnitude.
data = data[['Date', 'Time', 'Latitude', 'Longitude', 'Depth', 'Magnitude']]
data.head()
from mpl_toolkits.basemap import Basemap
m = Basemap(projection='mill',llcrnrlat=-80,urcrnrlat=80, llcrnrlon=-
180,urcrnrlon=180,lat_ts=20,resolution='c')
longitudes = data["Longitude"].tolist()
latitudes = data["Latitude"].tolist()
#m = Basemap(width=12000000,height=9000000,projection='lcc',
# resolution=None,lat_1=80.,lat_2=55,lat_0=80,lon_0=-107.)
x,y = m(longitudes,latitudes)
fig = plt.figure(figsize=(12,10))
plt.title("All affected areas")
m.plot(x, y, "o", markersize = 2, color = 'blue')
m.drawcoastlines()
m.fillcontinents(color='coral',lake_color='aqua')
m.drawmapboundary()
m.drawcountries()
plt.show()
Data
# demonstrate that the train-test split procedure is repeatable
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
# create dataset
X, y = make_blobs(n_samples=100)
# split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# summarize first 5 rows
print(X_train[:5, :])
# split again, and we should see the same split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
# summarize first 5 rows
print(X_train[:5, :])
[[-2.54341511 4.98947608]
[ 5.65996724 -8.50997751]
[-2.5072835 10.06155749]
[ 6.92679558 -5.91095498]
[ 6.01313957 -7.7749444 ]]
[[-2.54341511 4.98947608]
[ 5.65996724 -8.50997751]
[-2.5072835 10.06155749]
[ 6.92679558 -5.91095498]
[ 6.01313957 -7.7749444 ]]
# build a RandomForestClassifier
clf = RandomForestClassifier(n_estimators=50)
Specifying the list of parameters and distributions
param_dist = {"max_depth": [3, None],
"max_features": sp_randint(1, 11),
"min_samples_split": sp_randint(2, 11),
"min_samples_leaf": sp_randint(1, 11),
"bootstrap": [True, False],
"criterion": ["gini", "entropy"]}
Defining the sample, distributions and cross-validation
samples = 8 # number of random samples
randomCV = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=samples,cv=3)
All parameters are set and, let’s do the fit model
randomCV.fit(X, y)
print(randomCV.best_params_)
X = final_data[['Timestamp', 'Latitude', 'Longitude']]
y = final_data[['Magnitude', 'Depth']]
In [11]:
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, X_test.shape)
(18727, 3) (4682, 3) (18727, 2) (4682, 3) from sklearn.ensemble import RandomForestRegressor reg = RandomForestRegressor(random_state=42) reg.fit(X_train, y_train) reg.predict(X_test)
array([[ 5.96, 50.97],
[ 5.88, 37.8 ],
[ 5.97, 37.6 ],
...,
[ 6.42, 19.9 ],
[ 5.73, 591.55],
[ 5.68, 33.61]])
reg.score(X_test, y_test)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))
from mpl_toolkits.basemap import Basemap
m = Basemap(projection='mill',llcrnrlat=-80,urcrnrlat=80, llcrnrlon=-180,urcrnrlon=180,lat_ts=20,resolution='c')
longitudes = data["Longitude"].tolist()
latitudes = data["Latitude"].tolist()
#m = Basemap(width=12000000,height=9000000,projection='lcc',
#resolution=None,lat_1=80.,lat_2=55,lat_0=80,lon_0=-107.)
x,y = m(longitudes,latitudes)
In [9]: linkcode
fig = plt.figure(figsize=(12,10))
plt.title("All affected areas")
m.plot(x, y, "o", markersize = 2, color = 'blue')
m.drawcoastlines()
m.fillcontinents(color='coral',lake_color='aqua')
m.drawmapboundary()
m.drawcountries()
plt.show()
import datetime import time timestamp = [] for d, t in zip(data['Date'], data['Time']): try: ts = datetime.datetime.strptime(d+' '+t, '%m/%d/%Y %H:%M:%S') timestamp.append(time.mktime(ts.timetuple())) except ValueError: # print('ValueError') timestamp.append('ValueError')
timeStamp = pd.Series(timestamp) data['Timestamp'] = timeStamp.values final_data = data.drop(['Date', 'Time'], axis=1) final_data = final_data[final_data.Timestamp != 'ValueError']
final_data.head()
def gen_features(X):
strain=[] strain.append(X.mean()) strain.append(X.std()) strain.append(X.min()) strain.append(X.kurtosis()) strain.append(X.skew()) strain.append(np.quantile(X, 0.01)) return
pd.Series(strain) train = pd.read_csv('train.csv', iterator=True, chunksize=150_000, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})
X_train = pd.DataFrame() y_train = pd.Series() for df in train:
ch = gen_features(df['acoustic_data'])
X_train = X_train.append(ch, ignore_index=True) y_train = y_train.append(pd.Series(df['time_to_failure'].values[-1]))
scaler = StandardScaler() scaler.fit(X_train) X_train_scaled = scaler.transform(X_train) parameters = [{'gamma': [0.001, 0.005, 0.01, 0.02, 0.05, 0.1],
'C': [0.1, 0.2, 0.25, 0.5, 1, 1.5, 2]}]
#'nu': [0.75, 0.8, 0.85, 0.9, 0.95, 0.97]}] reg1 =
GridSearchCV(SVR(kernel='rbf', tol=0.01), parameters, cv=5,
scoring='neg_mean_absolute_error') reg1.fit(X_train_scaled, y_train.values.flatten()) y_pred1 = reg1.predict(X_train_scaled) print("Best CV score:
{:.4f}".format(reg1.best_score_)) print(reg1.best_params_)
parameters = [{'gamma': np.linspace(0.001, 0.1, 10),
'alpha': [0.005, 0.01, 0.02, 0.05, 0.1]}] reg2 =
GridSearchCV(KernelRidge(kernel='rbf'), parameters, cv=5,
scoring='neg_mean_absolute_error') reg2.fit(X_train_scaled, y_train.values.flatten()) y_pred2 = reg2.predict(X_train_scaled) print("Best CV score:
{:.4f}".format(reg2.best_score_)) print(reg2.best_params_)
plt.figure(figsize=(20, 5)) plt.plot(y_pred1,
color='green', label='SVR') plt.plot(y_pred2,
color='green', label='SVR') plt.plot(y_pred2,
color='orange', label='KernelRidge') plt.legend() plt.title('Kernel Ridge predictions vs
SVR predictions') plt.show()
port numpy as np
from sklearn.model_selection import train_test_split from sklearn.linear_model import LinearRegression from sklearn.metrics import mean_squared_error, r2_score
# Select the features we want to use
X = df[['latitude', 'longitude', 'depth', 'gap']] y
= df['mag']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Fit the model on the training set model = LinearRegression() model.fit(X_train, y_train)
# Evaluate the model on the testing set y_pred = model.predict(X_test) mse = mean_squared_error(y_test, y_pred) r2 = r2_score(y_test, y_pred) print('Mean squared error:', mse) print('R-squared score:', r2)
import matplotlib.pyplot as plt
plt.scatter(df['depth'], df['mag']) plt.xlabel('Depth') plt.ylabel('Magnitude') plt.title('Depth vs Magnitude') plt.show()
import matplotlib.pyplot as plt import seaborn as sns
# Scatter plot of magnitude vs. latitude plt.scatter(df['latitude'], df['mag'], alpha=0.2) plt.xlabel('Latitude') plt.ylabel('Magnitude') plt.title('Magnitude vs. Latitude')
plt.show()
# Scatter plot of magnitude vs. longitude plt.scatter(df['longitude'], df['mag'], alpha=0.2) plt.xlabel('Longitude') plt.ylabel('Magnitude') plt.title('Magnitude vs. Longitude') plt.show()
mean_mag_by_type = df.groupby('type')['mag'].mean(
.sort_values() mean_mag_by_type.plot(kind='barh', figsize=(10,6))
plt.title('Mean Magnitude by Earthquake Type')
plt.xlabel('Magnitude') plt.ylabel('Earthquake Type') plt.show()




