import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
# from PolyLearner import PolyLearner as Learner
from CARTLearner import CARTLearner

data = np.genfromtxt("tree_data/Istanbul.csv", delimiter=",")[1:,1:] # Load the Istanbul.csv file and remove the header row and date column.
# data = np.genfromtxt("tree_data/winequality-red.csv", delimiter=",")
# data = np.genfromtxt("tree_data/winequality-white.csv", delimiter=",")
# data = np.genfromtxt("tree_data/ripple.csv", delimiter=",")
# data = np.genfromtxt("tree_data/simple.csv", delimiter=",")
# data = np.genfromtxt("tree_data/3_groups.csv", delimiter=",")[1:,1:]



# Shuffle the rows and partition some data for testing.
x_train, x_test, y_train, y_test = train_test_split(data[:,:-1], data[:,-1], test_size=0.4)


learner = CARTLearner(leaf_size=1)
learner.train(x_train, y_train)

y_pred = learner.test(x_train)

# Test in-sample.
rmse_is = mean_squared_error(y_train, y_pred, squared=False)
corr_is = np.corrcoef(y_train, y_pred)[0,1]

# Test out-of-sample.
y_pred = learner.test(x_test)

rmse_oos = mean_squared_error(y_test, y_pred, squared=False)
corr_oos = np.corrcoef(y_test, y_pred)[0,1]

# Print summary.
print (f"In sample, RMSE: {rmse_is}, Corr: {corr_is}")
print (f"Out of sample, RMSE: {rmse_oos}, Corr: {corr_oos}")







