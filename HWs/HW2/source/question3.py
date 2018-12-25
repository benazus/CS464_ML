from sklearn import svm
import numpy as np

data = np.asarray(np.loadtxt("../data/UCI_Breast_Cancer.csv", dtype = "int32", delimiter=','))[:, 1:]
labels = data[:, -1]
data = data[:, :-1]

# 3.3
train_data = data[:500,:]
train_labels = labels[:500]
test_data = data[500:,:]
test_labels = labels[500:]
kf_test_data = np.asarray((train_data[:100], train_data[100:200], train_data[200:300], train_data[300:400], train_data[400:]))
kf_train_data = np.asarray((train_data[100:], np.concatenate((train_data[:100], train_data[200:])), np.concatenate((train_data[:200], train_data[300:])), np.concatenate((train_data[:300], train_data[400:])), train_data[:400]))
kf_test_labels = np.asarray((train_labels[:100], train_labels[100:200], train_labels[200:300], train_labels[300:400], train_labels[400:]))
kf_train_labels = np.asarray((train_labels[100:], np.concatenate((train_labels[:100], train_labels[200:])), np.concatenate((train_labels[:200], train_labels[300:])), np.concatenate((train_labels[:300], train_labels[400:])), train_labels[:400]))
C = np.array((10**-3, 10**-2, 10**-1, 1, 10**1, 10**2, 10**3))
errors = np.zeros(len(C))
index = 0
for c in C:
    mse = np.zeros(5)
    for i in range(5): # kfold
        model = svm.LinearSVC(C=c, dual=False, fit_intercept=True, loss = "squared_hinge", max_iter=1000, penalty = "l2", random_state=0, tol=1e-05)
        model.fit(kf_train_data[i], kf_train_labels[i])
        predictions = model.predict(kf_test_data[i])
        mse[i] = np.sum(np.power(kf_test_labels[i] - predictions, 2))
    errors[index] = float(np.sum(mse) / 5)
    index = index + 1

c = C[np.argmin(errors)]
print("Linear SVM with C = " + str(c))

model = svm.LinearSVC(C=c, dual=False, fit_intercept=True, loss = "squared_hinge", max_iter=1000, penalty = "l2", random_state=0, tol=1e-05)
model.fit(train_data, train_labels)
score = model.score(test_data, test_labels)
predictions = model.predict(test_data)
confusion_matrix = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}

for i in range(len(test_labels)):
    if test_labels[i] == 4 and predictions[i] == 4: # tp
        confusion_matrix["tp"] = confusion_matrix["tp"] + 1
    elif test_labels[i] == 2 and predictions[i] == 2: # tn
        confusion_matrix["tn"] = confusion_matrix["tn"] + 1
    elif test_labels[i] == 2 and predictions[i] == 4: # fp
        confusion_matrix["fp"] = confusion_matrix["fp"] + 1
    elif test_labels[i] == 4 and predictions[i] == 2: # fn
        confusion_matrix["fn"] = confusion_matrix["fn"] + 1
print("Confusion Matrix -> ")
print(confusion_matrix)
print("Accuracy: " + str(score))

# 3.4
train_data = data[:500,:]
train_labels = labels[:500]
test_data = data[500:,:]
test_labels = labels[500:]
kf_test_data = np.asarray((train_data[:100], train_data[100:200], train_data[200:300], train_data[300:400], train_data[400:]))
kf_train_data = np.asarray((train_data[100:], np.concatenate((train_data[:100], train_data[200:])), np.concatenate((train_data[:200], train_data[300:])), np.concatenate((train_data[:300], train_data[400:])), train_data[:400]))
kf_test_labels = np.asarray((train_labels[:100], train_labels[100:200], train_labels[200:300], train_labels[300:400], train_labels[400:]))
kf_train_labels = np.asarray((train_labels[100:], np.concatenate((train_labels[:100], train_labels[200:])), np.concatenate((train_labels[:200], train_labels[300:])), np.concatenate((train_labels[:300], train_labels[400:])), train_labels[:400]))
gammA = np.array((2**-4, 2**-3, 2**-2, 2**-1, 1, 2**1, 2**2, 2**3, 2**4))
errors = np.zeros(len(gammA))
index = 0
for gamma in gammA:
    mse = np.zeros(5)
    for i in range(5): # kfold
        model = svm.SVC(kernel = "rbf", gamma = gamma)
        model.fit(kf_train_data[i], kf_train_labels[i])
        predictions = model.predict(kf_test_data[i])
        mse[i] = np.sum(np.power(kf_test_labels[i] - predictions, 2))
    errors[index] = float(np.sum(mse) / 5)
    index = index + 1

gamma = gammA[np.argmin(errors)]
print("SVM with RBF kernel, gamma = " + str(gamma))

model = svm.SVC(kernel = "rbf", gamma = gamma)
model.fit(train_data, train_labels)
score = model.score(test_data, test_labels)
predictions = model.predict(test_data)
confusion_matrix = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}

for i in range(len(test_labels)):
    if test_labels[i] == 4 and predictions[i] == 4: # tp
        confusion_matrix["tp"] = confusion_matrix["tp"] + 1
    elif test_labels[i] == 2 and predictions[i] == 2: # tn
        confusion_matrix["tn"] = confusion_matrix["tn"] + 1
    elif test_labels[i] == 2 and predictions[i] == 4: # fp
        confusion_matrix["fp"] = confusion_matrix["fp"] + 1
    elif test_labels[i] == 4 and predictions[i] == 2: # fn
        confusion_matrix["fn"] = confusion_matrix["fn"] + 1
print("Confusion Matrix -> ")
print(confusion_matrix)
print("Accuracy: " + str(score))