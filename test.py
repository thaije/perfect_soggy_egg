import pandas as pd 
from sklearn import datasets, linear_model
import numpy as np
import os 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

f = "/home/minim/Projects/Fedya-chatbot/data/skills/soggy_egg_timer/egg_data.csv"


# contains data on the weight (and how long they should be cooked (target)
# f = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data.csv")
eggs = pd.read_csv(f, sep=",")

# normalize rating from 0-10 to 0-1
eggs['rating_normalized'] = eggs['rating'] / 10

# normalize the weight feature
scaler1 = MinMaxScaler()
wgr = [[x] for x in eggs['weight_gr'].to_list()]
print(wgr)
scaler1.fit(wgr)
eggs['weight_gr_normalized'] = scaler1.transform(wgr)

# normalize the cook seconds feature
scaler2 = MinMaxScaler()
cs = [[x] for x in eggs['cook_seconds'].to_list()]
scaler2.fit(cs)
eggs['cook_seconds_normalized'] = scaler2.transform(cs)


print(eggs.head())



# prepare axes for the plot 
fig, ax1 = plt.subplots()
ax1.set_xlabel('Cooking duration (seconds)')
ax1.set_ylabel('Egg weight (gram)')
ax1.set_title('The perfect soggy egg')

c = eggs.rating
plt.scatter( eggs['cook_seconds'], eggs['weight_gr'] , c=c, cmap = 'RdYlGn', s=70) 
# plot axes etc
ax1.set_ylim((40, 75))
ax1.set_xlim((300, 480))
cbar = plt.colorbar()
cbar.set_label('How great was the egg?')



# split the eggs dataframe into a training and test set
train = eggs.sample(frac=0.8, random_state=200)
test = eggs.drop(train.index)

print(train)
print(test)

regr = linear_model.LinearRegression()

# fit regr on two features: weight and rating, and predict the cooking time
regr.fit(train[['weight_gr_normalized', 'rating_normalized']], train['cook_seconds_normalized'])
print("Model coefficients:", regr.coef_)


real = test['cook_seconds_normalized'].to_list()
predicted = regr.predict(test[['weight_gr_normalized', 'rating_normalized']]).tolist()
mse = mean_squared_error(real, predicted)

# calc the mean squared error on the test set 
print("Metrics. Mean Squared Error:", mse)

# plot the learned linear function
test_y = [[item, 1] for item in list(range(40, 75, 5))]


def calc_datapoint(x):
    x = scaler1.transform([[x]])[0][0]
    y = regr.predict([[x, 1]])[0]
    y = scaler2.inverse_transform([[y]])[0][0]
    return y


# normalize for the linear regression model
test_y_normalized = []
for item in test_y:
    print(item)
    item_scaled = scaler1.transform([[item[0]]])[0][0]
    test_y_normalized.append([item_scaled, item[1]])

# calc predicated value  of test set
predicted_normalized = regr.predict(test_y_normalized)
# denormalize so we can plot it
predicted = scaler2.inverse_transform([[item] for item in predicted_normalized])
predicted = [item[0] for item in predicted]

# print("test_y_normalized", test_y_normalized)

print("Time for an egg of 61 gr:", calc_datapoint(61), "seconds")

plt.plot(predicted, [item[0] for item in test_y], color='blue', linewidth=3)

plt.show()
