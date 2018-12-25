import numpy as np
import matplotlib.pyplot as plt
import sys

def predict(w0, weights, sample):
    tmp = 1 / (1 + np.exp(w0 + np.sum(np.multiply(weights, sample))))
    return 0 if tmp > 0.5 else 1

def gradientAscent(lr, iteration_count, feature_count, sample_count, train_data, train_labels):
    w0 = 0
    weights = np.zeros((feature_count))
    for it in range(iteration_count):
        y_minus_prediction = np.asarray([train_labels[i] - predict(w0, weights, train_data[i]) for i in range(sample_count)])
        w0 = w0 + lr * np.sum(y_minus_prediction)
        tmp = train_data * y_minus_prediction[:, np.newaxis]
        tmp = tmp.sum(axis = 0)
        weights = weights + lr * tmp
    return (w0, weights)

def forwardSelection(kf_train_data, kf_train_labels, ic, lr):
    indices = np.asarray([0])
    previous_score = float(sys.maxsize)
    (t1, t2) = kf_train_data[0].shape
    for j in range(1, t2): # for each feature
        print("Feature at index " + str(j) + "...")
        new_indices = np.append(indices, j)
        mse = np.zeros(len(kf_train_data))

        for l in range(len(kf_train_data)): # k-fold
            new_data = kf_train_data[l][:, new_indices]
            new_test_data = kf_test_data[l][:, new_indices]
            (x_train, y_train) = new_data.shape
            (w0, weights) = gradientAscent(lr, ic, y_train, x_train, new_data, kf_train_labels[l])
            predictions = np.vectorize(lambda x: 0 if x > 0.5 else 1)(1 / (1 + np.exp(np.multiply(new_test_data, weights).sum(axis = 1) + w0)))
            mse[l] = np.sum(np.power(kf_test_labels[l] - predictions, 2))

        avg_mse = np.sum(mse) * float(1 / len(mse))
        print("avg_mse: " + str(avg_mse))
        print("previous_score: " + str(previous_score))
        if avg_mse < previous_score:
            print("Feature at index " + str(j) + " is accepted.")
            indices = new_indices 
            previous_score = avg_mse
        else:
            print("Feature at index " + str(j) + " is rejected.")
    return indices


def backwardElimination(kf_train_data, kf_train_labels, ic, lr):
    indices = np.arange(len(kf_train_data[0][0]))
    previous_score = -sys.maxsize
    (t1, t2) = kf_train_data[0].shape
    for i in range(t2):
        print("Feature at index " + str(i) + "...")
        new_indices = np.delete(indices, i)
        mse = np.zeros(len(kf_train_data))

        for l in range(len(kf_train_data)): # k-fold
            new_data = kf_train_data[l][:, new_indices]
            new_test_data = kf_test_data[l][:, new_indices]
            (x_train, y_train) = new_data.shape
            (w0, weights) = gradientAscent(lr, ic, y_train, x_train, new_data, kf_train_labels[l])
            predictions = np.vectorize(lambda x: 0 if x > 0.5 else 1)(1 / (1 + np.exp(np.multiply(new_test_data, weights).sum(axis = 1) + w0)))
            mse[l] = np.sum(np.power(kf_test_labels[l] - predictions, 2))

        avg_mse = np.sum(mse) * float(1 / len(mse))
        print("avg_mse: " + str(avg_mse))
        print("previous_score: " + str(previous_score))
        if avg_mse > previous_score:
            print("Feature at index " + str(i) + " is accepted.")
        else:
            print("Feature at index " + str(i) + " is rejected.")
            indices = new_indices 
            previous_score = avg_mse
    return indices

data = np.loadtxt("../data/ovariancancer.csv", dtype = "float", delimiter=',')
labels = np.loadtxt("../data/ovariancancer_labels.csv", dtype = "float", delimiter=',')
test_data = np.row_stack((data[:20,:], data[121:141,:]))
test_labels = np.concatenate((labels[:20], labels[121:141]))
train_data = np.row_stack((data[20:121,:], data[141:,:]))
train_labels = np.concatenate((labels[20:121], labels[141:]))
iteration_count = np.array([500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000])
learning_rate = np.array([0.001, 0.002, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03])
kf_test_data = np.asarray((train_data[:35], train_data[35:70], train_data[70:105], train_data[105:140], train_data[140:]))
kf_train_data = np.asarray((train_data[35:], np.concatenate((train_data[:35], train_data[70:])), np.concatenate((train_data[:70], train_data[105:])), np.concatenate((train_data[:105], train_data[140:])), train_data[:140]))
kf_test_labels = np.asarray((train_labels[:35], train_labels[35:70], train_labels[70:105], train_labels[105:140], train_labels[140:]))
kf_train_labels = np.asarray((train_labels[35:], np.concatenate((train_labels[:35], train_labels[70:])), np.concatenate((train_labels[:70], train_labels[105:])), np.concatenate((train_labels[:105], train_labels[140:])), train_labels[:140]))
kf_error_ic = np.zeros(len(iteration_count))
kf_error_lr = np.zeros(len(learning_rate))
index = 0

for ic in iteration_count:
    print("Iteration Count = " + str(ic))
    mse = np.zeros(len(kf_train_data))
    for i in range(len(kf_train_data)):
        print("k-Fold iteration " + str(i) + "...")
        (x_train, y_train) = kf_train_data[i].shape
        (x_test, y_test) = kf_test_data[i].shape
        (w0, weights) = gradientAscent(learning_rate[0], ic, y_train, x_train, kf_train_data[i], kf_train_labels[i])
        predictions = 1 / (1 + np.exp(np.multiply(kf_test_data[i], weights).sum(axis = 1) + w0))
        
        for k in range(len(predictions)):
            predictions[k] = 0 if predictions[k] > 0.5 else 1
            # if predictions[k] != kf_train_labels[i][k]:
            mse[i] = mse[i] + (kf_test_labels[i][k] - predictions[k]) ** 2
    kf_error_ic[index] = np.sum(mse) * 0.2
    index = index + 1

for lr in learning_rate:
    print("Learning Rate = " + str(lr))
    mse = np.zeros(len(kf_train_data))
    for i in range(len(kf_train_data)):
        print("k-Fold iteration " + str(i) + "...")
        (x_train, y_train) = kf_train_data[i].shape
        (x_test, y_test) = kf_test_data[i].shape
        (w0, weights) = gradientAscent(lr, iteration_count[0], y_train, x_train, kf_train_data[i], kf_train_labels[i])
        predictions = 1 / (1 + np.exp(np.multiply(kf_test_data[i], weights).sum(axis = 1) + w0))
        
        for k in range(len(predictions)):
            predictions[k] = 0 if predictions[k] > 0.5 else 1
            # if predictions[k] != kf_train_labels[i][k]:
            mse[i] = mse[i] + (kf_test_labels[i][k] - predictions[k]) ** 2
    kf_error_lr[index] = np.sum(mse) * 0.2
    index = index + 1
    
# k-Fold Results
kfold_ic = iteration_count[np.argmin(kf_error_ic)]
kfold_lr = learning_rate[np.argmin(kf_error_lr)]
(kfold_sample_count, kfold_feature_count) = test_data.shape
(kfold_w0, kfold_weights) = gradientAscent(0.001, 2500, kfold_feature_count, kfold_sample_count, test_data, test_labels)
kfold_predictions = 1 / (1 + np.exp(np.multiply(test_data, kfold_weights).sum(axis = 1) + kfold_w0))
confusion_matrix = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}

for i in range(len(kfold_predictions)):
    kfold_predictions[i] = 0 if kfold_predictions[i] > 0.5 else 1
    if test_labels[i] == 1 and kfold_predictions[i] == 1: # tp
        confusion_matrix["tp"] = confusion_matrix["tp"] + 1
    elif test_labels[i] == 0 and kfold_predictions[i] == 0: # tn
        confusion_matrix["tn"] = confusion_matrix["tn"] + 1
    elif test_labels[i] == 0 and kfold_predictions[i] == 1: # fp
        confusion_matrix["fp"] = confusion_matrix["fp"] + 1
    elif test_labels[i] == 1 and kfold_predictions[i] == 0: # fn
        confusion_matrix["fn"] = confusion_matrix["fn"] + 1

print("Iteration Count: " + str(kfold_ic) + ", Learning Rate: " + str(learning_rate))
print("Confusion Matrix: ")
print(confusion_matrix)

# Feature Selection
indices_forward = forwardSelection(kf_train_data, kf_train_labels, kfold_ic, kfold_lr)
print("Forward Selection...")
new_test_data = test_data[:, indices_forward]
(test_sample, test_feature) = new_test_data.shape
(w0, weights) = gradientAscent(kfold_lr, kfold_ic, test_feature, test_sample, new_test_data, test_labels)
predictions = 1 / (1 + np.exp(np.multiply(new_test_data, weights).sum(axis = 1) + w0))
confusion_matrix = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}

for i in range(len(predictions)):
    predictions[i] = 0 if predictions[i] > 0.5 else 1
    if test_labels[i] == 1 and predictions[i] == 1: # tp
        confusion_matrix["tp"] = confusion_matrix["tp"] + 1
    elif test_labels[i] == 0 and predictions[i] == 0: # tn
        confusion_matrix["tn"] = confusion_matrix["tn"] + 1
    elif test_labels[i] == 0 and predictions[i] == 1: # fp
        confusion_matrix["fp"] = confusion_matrix["fp"] + 1
    elif test_labels[i] == 1 and predictions[i] == 0: # fn
        confusion_matrix["fn"] = confusion_matrix["fn"] + 1
print("Confusion Matrix: ")
print(confusion_matrix)

# Backward Elimination
indices_backward = backwardElimination(kf_train_data, kf_train_labels, kfold_ic, kfold_lr)
print("Backward Elimination...")
new_test_data = test_data[:, indices_backward]
(test_sample, test_feature) = new_test_data.shape
(w0, weights) = gradientAscent(kfold_lr, kfold_ic, test_feature, test_sample, new_test_data, test_labels)
predictions = 1 / (1 + np.exp(np.multiply(new_test_data, weights).sum(axis = 1) + w0))
confusion_matrix = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}

for i in range(len(predictions)):
    predictions[i] = 0 if predictions[i] > 0.5 else 1
    if test_labels[i] == 1 and predictions[i] == 1: # tp
        confusion_matrix["tp"] = confusion_matrix["tp"] + 1
    elif test_labels[i] == 0 and predictions[i] == 0: # tn
        confusion_matrix["tn"] = confusion_matrix["tn"] + 1
    elif test_labels[i] == 0 and predictions[i] == 1: # fp
        confusion_matrix["fp"] = confusion_matrix["fp"] + 1
    elif test_labels[i] == 1 and predictions[i] == 0: # fn
        confusion_matrix["fn"] = confusion_matrix["fn"] + 1
print("Confusion Matrix: ")
print(confusion_matrix)