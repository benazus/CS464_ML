# Cylinders, Displacement, Horsepower, Weight, Acceleration, Model Year and MPG

import numpy as np
from numpy.linalg import inv, matrix_rank
import matplotlib.pyplot as plt

data = np.loadtxt("../data/carbig.csv", delimiter=',')
print("Question 1.2: ")
(x, y) = data.shape
train_sample_count = 300
train_data = data[:train_sample_count, : y - 1]
train_labels = data[:train_sample_count, y - 1]
test_data = data[train_sample_count : x + 1, : y - 1]
test_labels = data[train_sample_count: x + 1, y - 1]
# print(matrix_rank(np.transpose(train_data).dot(train_data)))
beta = inv(np.transpose(train_data).dot(train_data)).dot(np.transpose(train_data)).dot(train_labels)
train_predictions = train_data.dot(beta)
train_error = (np.square(train_labels - train_predictions)).mean(axis = None)
test_predictions = test_data.dot(beta)
test_error = (np.square(test_labels - test_predictions)).mean(axis = None)
print("Coefficients of Beta: ")
print(beta)
print("Train Error: " + str(train_error))
print("Test Error: " + str(test_error))

print("*******************************")
print("Question 1.5: ")
(x, y) = data.shape
plt.plot(data[:, 2], data[:, y - 1], ".")
plt.xlabel("Horsepower")
plt.ylabel("MPG")
plt.show()

print("*******************************")
print("Question 1.6: ")
features = np.column_stack((np.ones(x), data[:, 2], data[:, 2] ** 2, data[:, 2] ** 3, data[:, 2] ** 4, data[:, 2] ** 5))
f1 = features[:, 0]
f2 = features[:, 0:2]
f3 = features[:, 0:3]
f4 = features[:, 0:4]
f5 = features[:, 0:5]
f6 = features[:, 0:6]   
xTx1 = np.transpose(f1).dot(f1)
xTx2 = np.transpose(f2).dot(f2)
xTx3 = np.transpose(f3).dot(f3)
xTx4 = np.transpose(f4).dot(f4)
xTx5 = np.transpose(f5).dot(f5)
xTx6 = np.transpose(f6).dot(f6)
rank1 = matrix_rank(xTx1)
rank2 = matrix_rank(xTx2)
rank3 = matrix_rank(xTx3)
rank4 = matrix_rank(xTx4)
rank5 = matrix_rank(xTx5)
rank6 = matrix_rank(xTx6)
print("Ranks before normalization: ")
print("Rank for p = 0: " + str(rank1))
print("Rank for p = 1: " + str(rank2))
print("Rank for p = 2: " + str(rank3))
print("Rank for p = 3: " + str(rank4))
print("Rank for p = 4: " + str(rank5))
print("Rank for p = 5: " + str(rank6))

mean = features.mean(0)
stddev = features.std(0)
for i in range(x):
	for j in range(1, y - 1):
		features[i][j] = (lambda x, y, z, i, j: float(x - y[j]) / z[j])(features[i][j], mean, stddev, i, j) 

f1 = features[:, 0]
f2 = features[:, 0:2]
f3 = features[:, 0:3]
f4 = features[:, 0:4]
f5 = features[:, 0:5]
f6 = features[:, 0:6]   
xTx1 = np.transpose(f1).dot(f1)
xTx2 = np.transpose(f2).dot(f2)
xTx3 = np.transpose(f3).dot(f3)
xTx4 = np.transpose(f4).dot(f4)
xTx5 = np.transpose(f5).dot(f5)
xTx6 = np.transpose(f6).dot(f6)
rank1 = matrix_rank(xTx1)
rank2 = matrix_rank(xTx2)
rank3 = matrix_rank(xTx3)
rank4 = matrix_rank(xTx4)
rank5 = matrix_rank(xTx5)
rank6 = matrix_rank(xTx6)
print("Ranks before normalization: ")
print("Rank for p = 0: " + str(rank1))
print("Rank for p = 1: " + str(rank2))
print("Rank for p = 2: " + str(rank3))
print("Rank for p = 3: " + str(rank4))
print("Rank for p = 4: " + str(rank5))
print("Rank for p = 5: " + str(rank6))

print("*******************************")
print("Question 1.7: ")
hp = data[:, 2]
mean_hp = hp.mean(0)
stddev_hp = hp.std(0)
for i in range(x):
    hp[i] = (lambda x: (x - mean_hp) / stddev_hp)(hp[i])

features_17_1 = np.column_stack((np.ones(x), hp))
features_17_2 = np.column_stack((np.ones(x), hp, hp ** 2))
features_17_3 = np.column_stack((np.ones(x), hp, hp ** 2, hp ** 3))
features_17_4 = np.column_stack((np.ones(x), hp, hp ** 2, hp ** 3, hp ** 4))
features_17_5 = np.column_stack((np.ones(x), hp, hp ** 2, hp ** 3, hp ** 4, hp ** 5))

# p = 1
train_sample_count = 300
train_data = features_17_1[:train_sample_count, : y - 1]
train_labels = data[:train_sample_count, y - 1]
test_data = features_17_1[train_sample_count : x + 1, : y - 1]
test_labels = data[train_sample_count: x + 1, y - 1]
beta_1 = inv(np.transpose(train_data).dot(train_data)).dot(np.transpose(train_data)).dot(train_labels)
train_predictions = train_data.dot(beta_1)
train_error = (np.square(train_labels - train_predictions)).mean(axis = None)
test_predictions = test_data.dot(beta_1)
test_error = (np.square(test_labels - test_predictions)).mean(axis = None)
print("Coefficients of Beta for p = 1: ")
print(beta_1)
print("Train Error for p = 1: " + str(train_error))
print("Test Error for p = 1: " + str(test_error))

# p = 2
train_sample_count = 300
train_data = features_17_2[:train_sample_count, : y - 1]
train_labels = data[:train_sample_count, y - 1]
test_data = features_17_2[train_sample_count : x + 1, : y - 1]
test_labels = data[train_sample_count: x + 1, y - 1]
beta_2 = inv(np.transpose(train_data).dot(train_data)).dot(np.transpose(train_data)).dot(train_labels)
train_predictions = train_data.dot(beta_2)
train_error = (np.square(train_labels - train_predictions)).mean(axis = None)
test_predictions = test_data.dot(beta_2)
test_error = (np.square(test_labels - test_predictions)).mean(axis = None)
print("Coefficients of Beta for p = 2: ")
print(beta_2)
print("Train Error for p = 2: " + str(train_error))
print("Test Error for p = 2: " + str(test_error))

# p = 3
train_sample_count = 300
train_data = features_17_3[:train_sample_count, : y - 1]
train_labels = data[:train_sample_count, y - 1]
test_data = features_17_3[train_sample_count : x + 1, : y - 1]
test_labels = data[train_sample_count: x + 1, y - 1]
beta_3 = inv(np.transpose(train_data).dot(train_data)).dot(np.transpose(train_data)).dot(train_labels)
train_predictions = train_data.dot(beta_3)
train_error = (np.square(train_labels - train_predictions)).mean(axis = None)
test_predictions = test_data.dot(beta_3)
test_error = (np.square(test_labels - test_predictions)).mean(axis = None)
print("Coefficients of Beta for p = 3: ")
print(beta_3)
print("Train Error for p = 3: " + str(train_error))
print("Test Error for p = 3: " + str(test_error))

# p = 4
train_sample_count = 300
train_data = features_17_4[:train_sample_count, : y - 1]
train_labels = data[:train_sample_count, y - 1]
test_data = features_17_4[train_sample_count : x + 1, : y - 1]
test_labels = data[train_sample_count: x + 1, y - 1]
beta_4 = inv(np.transpose(train_data).dot(train_data)).dot(np.transpose(train_data)).dot(train_labels)
train_predictions = train_data.dot(beta_4)
train_error = (np.square(train_labels - train_predictions)).mean(axis = None)
test_predictions = test_data.dot(beta_4)
test_error = (np.square(test_labels - test_predictions)).mean(axis = None)
print("Coefficients of Beta for p = 4: ")
print(beta_4)
print("Train Error for p = 4: " + str(train_error))
print("Test Error for p = 4: " + str(test_error))

# p = 5
train_sample_count = 300
train_data = features_17_5[:train_sample_count, : y - 1]
train_labels = data[:train_sample_count, y - 1]
test_data = features_17_5[train_sample_count : x + 1, : y - 1]
test_labels = data[train_sample_count: x + 1, y - 1]
beta_5 = inv(np.transpose(train_data).dot(train_data)).dot(np.transpose(train_data)).dot(train_labels)
train_predictions = train_data.dot(beta_5)
train_error = (np.square(train_labels - train_predictions)).mean(axis = None)
test_predictions = test_data.dot(beta_5)
test_error = (np.square(test_labels - test_predictions)).mean(axis = None)
print("Coefficients of Beta for p = 5: ")
print(beta_5)
print("Train Error for p = 5: " + str(train_error))
print("Test Error for p = 5: " + str(test_error))

shp = np.sort(hp)
plt.plot(shp, (lambda x: beta_1[0] + beta_1[1] * x)(shp), "r", label = "p = 1")
plt.plot(shp, (lambda x: beta_2[0] + beta_2[1] * x + beta_2[2] * (x ** 2))(shp), "g", label = "p = 2")
plt.plot(shp, (lambda x: beta_3[0] + beta_3[1] * x + beta_3[2] * (x ** 2) + beta_3[3] * (x ** 3))(shp), "b", label = "p = 3")
plt.plot(shp, (lambda x: beta_4[0] + beta_4[1] * x + beta_4[2] * (x ** 2) + beta_4[3] * (x ** 3) + beta_4[4] * (x ** 4))(shp), "m", label = "p = 4")
plt.plot(shp, (lambda x: beta_5[0] + beta_5[1] * x + beta_5[2] * (x ** 2) + beta_5[3] * (x ** 3) + beta_5[4] * (x ** 4) + beta_5[5] * (x ** 5))(shp), "y", label = "p = 5")
plt.legend()
plt.show()

print("*******************************")
print("Question 1.8: ")
my = data[:, 5]
mean_my = my.mean(0)
stddev_my = my.std(0)
for i in range(x):
    my[i] = (lambda x: (x - mean_my) / stddev_my)(my[i])

features_18_1 = np.column_stack((np.ones(x), hp, my))
features_18_2 = np.column_stack((np.ones(x), hp, hp ** 2, my, my ** 2))
features_18_3 = np.column_stack((np.ones(x), hp, hp ** 2, hp ** 3, my, my ** 2, my ** 3))

# p = 1
train_sample_count = 300
train_data = features_18_1[:train_sample_count, : y - 1]
train_labels = data[:train_sample_count, y - 1]
test_data = features_18_1[train_sample_count : x + 1, : y - 1]
test_labels = data[train_sample_count: x + 1, y - 1]
beta = inv(np.transpose(train_data).dot(train_data)).dot(np.transpose(train_data)).dot(train_labels)
train_predictions = train_data.dot(beta)
train_error = (np.square(train_labels - train_predictions)).mean(axis = None)
test_predictions = test_data.dot(beta)
test_error = (np.square(test_labels - test_predictions)).mean(axis = None)
print("Coefficients of Beta for p = 1: ")
print(beta)
print("Train Error for p = 1: " + str(train_error))
print("Test Error for p = 1: " + str(test_error))

# p = 2
train_sample_count = 300
train_data = features_18_2[:train_sample_count, : y - 1]
train_labels = data[:train_sample_count, y - 1]
test_data = features_18_2[train_sample_count : x + 1, : y - 1]
test_labels = data[train_sample_count: x + 1, y - 1]
beta = inv(np.transpose(train_data).dot(train_data)).dot(np.transpose(train_data)).dot(train_labels)
train_predictions = train_data.dot(beta)
train_error = (np.square(train_labels - train_predictions)).mean(axis = None)
test_predictions = test_data.dot(beta)
test_error = (np.square(test_labels - test_predictions)).mean(axis = None)
print("Coefficients of Beta for p = 2: ")
print(beta)
print("Train Error for p = 2: " + str(train_error))
print("Test Error for p = 2: " + str(test_error))

# p = 3
train_sample_count = 300
train_data = features_18_3[:train_sample_count, : y - 1]
train_labels = data[:train_sample_count, y - 1]
test_data = features_18_3[train_sample_count : x + 1, : y - 1]
test_labels = data[train_sample_count: x + 1, y - 1]
beta = inv(np.transpose(train_data).dot(train_data)).dot(np.transpose(train_data)).dot(train_labels)
train_predictions = train_data.dot(beta)
train_error = (np.square(train_labels - train_predictions)).mean(axis = None)
test_predictions = test_data.dot(beta)
test_error = (np.square(test_labels - test_predictions)).mean(axis = None)
print("Coefficients of Beta for p = 3: ")
print(beta)
print("Train Error for p = 3: " + str(train_error))
print("Test Error for p = 3: " + str(test_error))