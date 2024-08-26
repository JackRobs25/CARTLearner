The CARTLearner class implements a simple decision tree regression model based on the Classification and Regression Trees (CART) algorithm.

	•	Initialization: CARTLearner(leaf_size=1) initializes the learner with a specified leaf size, determining the minimum number of samples required to create a leaf node.
	•	Training: train(x, y) takes feature data x and target values y, concatenates them, and builds the decision tree using the build_tree method.
	•	Tree Building: build_tree(data) recursively constructs the decision tree. It splits the data based on the feature that maximizes correlation with the target and continues to build sub-trees until the stopping criteria (uniform labels, identical features, or leaf size) are met.
	•	Feature Selection: select_split(data) finds the feature and split value that results in the highest correlation with the target variable.
	•	Testing: test(x) predicts target values for new input data x using the trained decision tree.
	•	Prediction: predict_row(row, tree) traverses the decision tree to predict the target value for a single input row.

This implementation provides a basic decision tree model for regression tasks, focusing on correlation-based splits and handling various stopping conditions.
