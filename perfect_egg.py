# requires python3 with matplotlib, pandas, sklearn, nmumpy

import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
import numpy as np
import os

f = os.path.join(os.path.dirname(os.path.abspath(__file__)), "egg_data.csv")
eggs = pd.read_csv(f, sep=",")


# prepare axes for the plot 
fig, ax1 = plt.subplots()
ax1.set_xlabel('Kook duur (seconden)')
ax1.set_ylabel('Gewicht ei (gram)')
ax1.set_title('Het perfecte ei')

# get colours 
c = eggs.rating

# make the scatterplot 
plt.scatter( eggs['cook_seconds'], eggs['weight_gr'] , c=c, cmap = 'RdYlGn', s=70) 
ax1.set_ylim((40, 75))
ax1.set_xlim((300, 480))
cbar = plt.colorbar()
cbar.set_label('Hoe goed was het ei?')


# get the perfect stuff 
best_eggs = eggs[eggs['rating'] > 8] 
best_eggs = best_eggs.reset_index()
# print(best_eggs)

y = []
x = []
weights = best_eggs['rating'].to_list()
for i in range(best_eggs.shape[0]):
    y.append([best_eggs.loc[i,:]['weight_gr']])
    x.append(best_eggs.loc[i,:]['cook_seconds'])

# print(x)
# print(y)

# # print (best_eggs['cook_seconds'].to_list())

# fit a linear function to it 
regr = linear_model.LinearRegression()
regr.fit(y, x, weights)


test_y = [[item] for item in list(range(40, 75, 5))]



plt.plot(regr.predict(test_y), test_y, color='blue', linewidth=3)


plt.show()
