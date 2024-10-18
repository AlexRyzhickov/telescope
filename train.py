import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('data.csv')
df = df.drop(
    [
        'Azimuth',
        'Elevation',
        'Wind_Direction',
        'Pressure',
        'Inclinometer1_X',
        'Inclinometer1_Y',
        'Inclinometer2_X'
    ],
    axis=1
)

print(df.info())

num_list = list(df.columns)

fig = plt.figure(figsize=(10, 30))

for i in range(len(num_list)):
    plt.subplot(21, 2, i + 1)
    plt.title(num_list[i])
    plt.hist(df[num_list[i]], color='blue', alpha=0.5)

plt.tight_layout()
# plt.show()

# corr map
# sns.pairplot(df)
# plt.figure(figsize = (10,8))
# sns.heatmap(df.corr(),annot=True, cbar=False, cmap='Blues', fmt='.1f')

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

X = df.drop(['DeltaA', 'DeltaE', 'Roll'], axis=1)
Y1 = df[['DeltaA']]
Y2 = df[['DeltaE']]
Y3 = df[['Roll']]

X_train, X_test, y1_train, y1_test, y2_train, y2_test, y3_train, y3_test = train_test_split(X, Y1, Y2, Y3, test_size=0.33, random_state=20)

MinMax = MinMaxScaler(feature_range=(0, 1))
X_train = MinMax.fit_transform(X_train)
X_test = MinMax.transform(X_test)

model_1 = GradientBoostingRegressor(
    learning_rate=0.1,
    n_estimators=250,
    max_depth=5,
    min_samples_split=2,
    min_samples_leaf=3,
    subsample=1.0
)

model_2 = GradientBoostingRegressor(
    learning_rate=0.1,
    n_estimators=250,
    max_depth=5,
    min_samples_split=2,
    min_samples_leaf=3,
    subsample=1.0
)

model_3 = GradientBoostingRegressor(
    learning_rate=0.1,
    n_estimators=250,
    max_depth=5,
    min_samples_split=2,
    min_samples_leaf=3,
    subsample=1.0
)

model_1.fit(X_train, y1_train)
actr1 = r2_score(y1_train, model_1.predict(X_train))
y1_pred = model_1.predict(X_test)
acte1 = r2_score(y1_test, y1_pred)

model_2.fit(X_train, y2_train)
actr2 = r2_score(y2_train, model_2.predict(X_train))
y2_pred = model_2.predict(X_test)
acte2 = r2_score(y2_test, y2_pred)

model_3.fit(X_train, y3_train)
actr3 = r2_score(y3_train, model_3.predict(X_train))
y3_pred = model_3.predict(X_test)
acte3 = r2_score(y3_test, y3_pred)

print("GradientBoostingRegressor: R-Squared on Y1 train dataset={}".format(actr1))
print("GradientBoostingRegressor: R-Squared on Y1 test dataset={}".format(acte1))
print("GradientBoostingRegressor: R-Squared on Y2 train dataset={}".format(actr2))
print("GradientBoostingRegressor: R-Squared on Y2 test dataset={}".format(acte2))
print("GradientBoostingRegressor: R-Squared on Y3 train dataset={}".format(actr3))
print("GradientBoostingRegressor: R-Squared on Y3 test dataset={}".format(acte3))

x_ax = range(len(y1_test))
plt.figure(figsize=(20, 10))
plt.subplot(3, 1, 1)
plt.plot(x_ax, y1_test, label="Actual DeltaA")
plt.plot(x_ax, y1_pred, label="Predicted DeltaA")
plt.title("DeltaA test and predicted data")
plt.xlabel('X-axis')
plt.ylabel('DeltaA')
plt.legend(loc='best', fancybox=True, shadow=True)
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(x_ax, y2_test, label="Actual DeltaE")
plt.plot(x_ax, y2_pred, label="Predicted DeltaE")
plt.title("DeltaE test and predicted data")
plt.xlabel('X-axis')
plt.ylabel('DeltaE')
plt.legend(loc='best', fancybox=True, shadow=True)
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(x_ax, y3_test, label="Actual Roll")
plt.plot(x_ax, y3_pred, label="Predicted Roll")
plt.title("Roll test and predicted data")
plt.xlabel('X-axis')
plt.ylabel('Roll')
plt.legend(loc='best', fancybox=True, shadow=True)
plt.grid(True)

plt.show()

import joblib
joblib.dump(model_1, 'model_1.pkl')
joblib.dump(model_2, 'model_2.pkl')
joblib.dump(model_3, 'model_3.pkl')

