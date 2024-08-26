import numpy as np

class CARTLearner:
    def __init__(self, leaf_size=1):
        self.leaf_size = leaf_size
        self.tree = None

    def train(self, x, y):
        data = np.column_stack((x, y)) # concatenate the input features x and target values y into a single array data. Each row of this array represents a training example, where the last column contains the target values.
        self.tree = self.build_tree(data)

    def build_tree(self, data):
        features = data[:, :-1]
        labels = data[:, -1]
        if len(set(labels)) == 1:  # All y values are the same
            return {'leaf': True, 'value': data[0, -1]} # return that y value 

        if len(set(features.flatten())) == 1:  # all X values are same - can't split
            return {'leaf': True, 'value': np.mean(data[:, -1])} # return mean of remaining y values

        if features.shape[0] <= self.leaf_size:  # leaf_size reached
            return {'leaf': True, 'value': np.mean(data[:, -1])}

        # Determine the best X feature to split on
        best_feature, split_value = self.select_split(data)


        if best_feature is None or split_value is None:
            return {'leaf': True, 'value': np.mean(data[:, -1])}
        
        print("data: ", data)
        print("split_value: ", split_value)
        print("split_feature: ", best_feature)

        left_data = data[data[:, best_feature] <= split_value]
        right_data = data[data[:, best_feature] > split_value]

        print("left_data: ", left_data)
        print("right_data: ", right_data)

        # if we have a case where all features fall on one side of the split value we have to call it a day 
        if (len(left_data) == 0 or len(right_data) == 0):
            return {'leaf': True, 'value': np.mean(data[:, -1])}
        
        left_child = self.build_tree(left_data)
        right_child = self.build_tree(right_data)

        return {'leaf': False, 'feature': best_feature, 'split_value': split_value,
                'left_child': left_child, 'right_child': right_child}


    def select_split(self, data):
        features = data[:, :-1]
        target = data[:, -1]
        

        best_feature = None
        best_split_value = None
        max_correlation = 0.0

        for i in range(features.shape[1]): # for each column i.e ticker for a stock example
            feature_values = features[:, i] # all values for that feature (stock)
            correlation_matrix = np.corrcoef(feature_values, target)
            correlation_coefficient = correlation_matrix[0, 1]
            correlation = np.abs(correlation_coefficient)

            if correlation > max_correlation:
                max_correlation = correlation
                best_feature = i
                best_split_value = np.median(feature_values)

        return best_feature, best_split_value

    def test(self, x):
        predictions = []

        for row in x:
            predictions.append(self.predict_row(row, self.tree))

        return np.array(predictions)

    def predict_row(self, row, tree):
        if tree['leaf']:
            return tree['value']

        if row[tree['feature']] <= tree['split_value']:
            return self.predict_row(row, tree['left_child'])
        else:
            return self.predict_row(row, tree['right_child'])
