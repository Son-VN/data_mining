import numpy as np
from sklearn.metrics import accuracy_score

np.set_printoptions(precision=3)


def entropy(ar):
    classes, counts = np.unique(ar, return_counts=True)
    probabilities = counts / len(ar)
    e = -np.sum(
        probabilities * np.log2(probabilities + 1e-10)
    )  # Add a small epsilon to avoid log(0)
    return e


class DecisionStumpClassifier:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.left_class = None
        self.right_class = None

    def fit(self, X, y):
        best_entropy = float("inf")
        n_samples = len(y)

        for i in range(n_samples):
            if i == 0:
                X_pre = 0
            else:
                X_pre = X[i - 1]
            if X[i] != X_pre:  # Skip duplicates
                threshold = round((X[i] + X_pre) / 2, 2)  # Midpoint threshold
                left_indices = np.where(X <= threshold)[0]
                right_indices = np.where(X > threshold)[0]
                left_labels = y[left_indices]
                right_labels = y[right_indices]
                left_entropy = entropy(y[left_indices])
                right_entropy = entropy(y[right_indices])
                total_entropy = (len(left_indices) / n_samples) * left_entropy + (
                    len(right_indices) / n_samples
                ) * right_entropy

                if total_entropy < best_entropy:
                    best_entropy = total_entropy
                    self.threshold = threshold
                    self.left_class = self._get_majority_class(left_labels)
                    self.right_class = self._get_majority_class(right_labels)
                    if self.left_class is None:
                        self.left_class = self.right_class
                    if self.right_class is None:
                        self.right_class = self.left_class

    def predict(self, X):
        return np.where(X > self.threshold, self.right_class, self.left_class)

    def _calculate_entropy(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy = -np.sum(
            probabilities * np.log2(probabilities + 1e-10)
        )  # Thêm epsilon nhỏ để tránh log(0)
        return entropy

    def _get_majority_class(self, ar):
        if len(ar) == 0:
            return None
        unique, counts = np.unique(ar, return_counts=True)
        return unique[np.argmax(counts)]


def bootstrap_sample(X, y, n_samples, replace=True):
    # Create training set Di by sampling from D
    indices = np.random.choice(X.shape[0], size=n_samples, replace=replace)
    X_bootstrap = X[indices]
    y_bootstrap = y[indices]
    sorted_indices = np.argsort(X_bootstrap)
    X_bootstrap = X_bootstrap[sorted_indices]
    y_bootstrap = y_bootstrap[sorted_indices]

    return X_bootstrap, y_bootstrap


class AdaBoost:
    def __init__(self, base_classifier, n_estimators=50):
        self.base_classifier = base_classifier
        self.n_estimators = n_estimators  # equivalent to k
        self.alphas = []
        self.classifiers = []

    def fit(self, X, y, n_samples, debug=False):

        # Initialize the weights for all N examples
        weights = np.ones(n_samples) / n_samples

        for i in range(self.n_estimators):
            if debug:
                print("\n===========> Round {} : <===========".format(i + 1))
            X_bootstrap, y_bootstrap = bootstrap_sample(X, y, n_samples, replace=True)
            if debug:
                print("X: {}".format(X_bootstrap.reshape(1, -1)))

            error = 1
            while error > 0.5:
                # Create training set Di by sampling from D

                classifier = self.base_classifier()
                classifier.fit(X_bootstrap, y_bootstrap)

                predictions = classifier.predict(X)
                # Caculate the weighted error
                error = np.sum(weights * (predictions != y)) / n_samples

            if debug:
                print("Threshold: {}".format(classifier.threshold))
                print("Y left: {}".format(classifier.left_class))
                print("Y right: {}".format(classifier.right_class))
                # print("Error: {}".format(error))
            # caculate alpha
            alpha = 0.5 * np.log((1 - error) / (error + 1e-10))
            self.alphas.append(alpha)
            if debug:
                print("Alpha: {}".format(alpha))

            # Update weights
            weights *= np.exp(
                -alpha * y_bootstrap * predictions
            )  # y_bootstrap and prediction is array of 1 or -1
            weights /= np.sum(weights)

            self.classifiers.append(classifier)

            if debug:
                print("Weights: {}".format(weights))
                print("Y: {}".format(predictions))

    def predict(self, X, debug=False):
        predictions = np.zeros(X.shape[0])
        for alpha, classifier in zip(self.alphas, self.classifiers):
            predictions += alpha * classifier.predict(X)
        if debug:
            print("Sum: {}".format(predictions))
        return np.sign(predictions)


# Example usage:
if __name__ == "__main__":
    from sklearn.metrics import accuracy_score

    # Create a dataset
    X = np.array([6, 8, 1, 4, 5, 6, 3, 2, 5, 10])
    # X = X.reshape(-1, 1)
    y = np.array([1, -1, 1, 1, -1, 1, -1, -1, 1, -1])

    print("X: {}".format(X.reshape(1, -1)))
    print("Y: {}".format(y))

    # # Split the dataset
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=42
    # )

    adaboost = AdaBoost(base_classifier=DecisionStumpClassifier, n_estimators=10)
    adaboost.fit(X, y, n_samples=X.shape[0], debug=True)  # Training

    print("\n++++++++++ Result ++++++++++")
    print("Y True: {}".format(y))
    y_pred = adaboost.predict(X, debug=True)
    print("Y Prediction: {}".format(y_pred))
    # Evaluate accuracy
    accuracy = accuracy_score(y, y_pred)
    print(f"Accuracy: {accuracy}")


""" Output

X: [[0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1. ]]
Y: [ 1  1  1 -1 -1 -1 -1  1  1  1]

===========> Round 1 : <===========
X: [[0.1 0.1 0.3 0.5 0.5 0.5 0.6 0.8 1.  1. ]]
Threshold: 0.4
Y left: 1
Y right: -1
Alpha: 1.5890269139239728
Weights: [0.01  0.01  0.01  0.235 0.01  0.01  0.01  0.235 0.235 0.235]
Y: [ 1  1  1  1 -1 -1 -1 -1 -1 -1]

===========> Round 2 : <===========
X: [[0.1 0.1 0.4 0.4 0.5 0.6 0.8 0.9 1.  1. ]]
Threshold: 0.7
Y left: -1
Y right: 1
Alpha: 2.9130000355102843
Weights: [0.304 0.304 0.001 0.022 0.001 0.001 0.304 0.022 0.022 0.022]
Y: [-1 -1 -1 -1 -1 -1 -1  1  1  1]

===========> Round 3 : <===========
X: [[0.1 0.2 0.2 0.2 0.3 0.6 0.6 0.7 0.8 1. ]]
Threshold: 0.45
Y left: 1
Y right: -1
Alpha: 2.3735518195673113
Weights: [5.049e-02 5.049e-02 1.489e-04 3.574e-03 1.716e-02 1.489e-04 5.049e-02
 3.574e-03 4.120e-01 4.120e-01]
Y: [ 1  1  1  1 -1 -1 -1 -1 -1 -1]

===========> Round 4 : <===========
X: [[0.1 0.3 0.4 0.5 0.6 0.6 0.7 0.8 0.9 1. ]]
Threshold: 0.75
Y left: -1
Y right: 1
Alpha: 2.2918935295142853
Weights: [4.583e-01 4.583e-01 1.381e-05 3.315e-04 1.592e-03 1.381e-05 4.682e-03
 3.315e-04 3.820e-02 3.820e-02]
Y: [-1 -1 -1 -1 -1 -1 -1  1  1  1]

===========> Round 5 : <===========
X: [[0.1 0.2 0.2 0.3 0.3 0.5 0.5 0.6 0.6 0.7]]
Threshold: 0.4
Y left: 1
Y right: -1
Alpha: 2.4289457634105998
Weights: [3.809e-01 3.809e-01 1.148e-05 2.755e-04 1.703e-01 1.148e-05 3.891e-03
 2.755e-04 3.175e-02 3.175e-02]
Y: [ 1  1  1  1 -1 -1 -1 -1 -1 -1]

===========> Round 6 : <===========
X: [[0.1 0.2 0.3 0.3 0.3 0.5 0.6 0.6 0.7 1. ]]
Threshold: 0.4
Y left: 1
Y right: -1
Alpha: 2.5221594217412138
Weights: [1.185e-02 1.185e-02 3.571e-07 8.569e-06 8.220e-01 3.571e-07 1.210e-04
 8.569e-06 9.876e-04 1.532e-01]
Y: [ 1  1  1  1 -1 -1 -1 -1 -1 -1]

===========> Round 7 : <===========
X: [[0.3 0.3 0.3 0.3 0.4 0.4 0.5 0.7 0.7 0.7]]
Threshold: 0.35
Y left: 1
Y right: -1
Alpha: 2.078231803827561
Weights: [1.184e-02 1.184e-02 3.569e-07 5.468e-04 8.215e-01 3.569e-07 1.210e-04
 8.565e-06 9.871e-04 1.531e-01]
Y: [ 1  1  1 -1 -1 -1 -1 -1 -1 -1]

===========> Round 8 : <===========
X: [[0.2 0.3 0.4 0.4 0.6 0.7 0.7 0.8 0.9 0.9]]
Threshold: 0.75
Y left: -1
Y right: 1
Alpha: 3.0215899514182127
Weights: [4.554e-01 4.554e-01 3.258e-08 4.993e-05 7.501e-02 3.258e-08 1.105e-05
 7.820e-07 9.012e-05 1.398e-02]
Y: [-1 -1 -1 -1 -1 -1 -1  1  1  1]

===========> Round 9 : <===========
X: [[0.3 0.3 0.6 0.6 0.6 0.8 0.9 0.9 0.9 1. ]]
Threshold: 0.7
Y left: -1
Y right: 1
Alpha: 1.1502238432847294
Weights: [4.951e-01 4.951e-01 3.550e-09 5.440e-06 8.172e-03 3.542e-08 1.201e-05
 8.520e-08 9.819e-06 1.523e-03]
Y: [-1 -1 -1 -1 -1 -1 -1  1  1  1]

===========> Round 10 : <===========
X: [[0.1 0.2 0.2 0.3 0.4 0.6 0.6 0.7 0.7 0.8]]
Threshold: 0.35
Y left: 1
Y right: -1
Alpha: 4.391403224918156
Weights: [4.514e-02 4.514e-02 3.237e-10 3.234e-03 7.451e-04 3.230e-09 1.095e-06
 7.768e-09 8.953e-07 9.057e-01]
Y: [ 1  1  1 -1 -1 -1 -1 -1 -1 -1]

++++++++++ Result ++++++++++
Y True: [ 1  1  1 -1 -1 -1 -1  1  1  1]
Sum: [  6.007   6.007   6.007  -6.933 -24.76  -24.76  -24.76   -6.007  -6.007
  -6.007]
Y Prediction: [ 1.  1.  1. -1. -1. -1. -1. -1. -1. -1.]
Accuracy: 0.7
PS C:\Users\Vu Ngoc Son> & "C:/Users/Vu Ngoc Son/AppData/Local/Programs/Python/Python312/python.exe" d:/aaa/DataMining-main/adaboost.py
X: [[ 6  8  1  4  5  6  3  2  5 10]]
Y: [ 1 -1  1  1 -1  1 -1 -1  1 -1]

===========> Round 1 : <===========
X: [[1 1 1 3 3 4 4 5 6 8]]
Threshold: 2.0
Y left: 1
Y right: -1
Alpha: 1.47221948858322
Weights: [0.161 0.161 0.008 0.008 0.008 0.161 0.161 0.161 0.161 0.008]
Y: [-1 -1  1 -1 -1 -1 -1  1 -1 -1]

===========> Round 2 : <===========
X: [[ 1  1  1  2  2  3  5  5  5 10]]
Threshold: 1.5
Y left: 1
Y right: -1
Alpha: 1.4812125367683877
Weights: [0.243 0.243 0.001 0.001 0.001 0.013 0.243 0.013 0.243 0.001]
Y: [-1 -1  1 -1 -1 -1 -1 -1 -1 -1]

===========> Round 3 : <===========
X: [[1 1 3 3 3 4 5 5 5 6]]
Threshold: 2.0
Y left: 1
Y right: -1
Alpha: 1.4598289572871788
Weights: [4.612e-01 4.612e-01 1.255e-03 6.770e-05 6.770e-05 2.384e-02 2.488e-02
 1.286e-03 2.488e-02 1.255e-03]
Y: [-1 -1  1 -1 -1 -1 -1  1 -1 -1]

===========> Round 4 : <===========
X: [[ 1  2  4  4  5  8  8  8 10 10]]
Threshold: 4.5
Y left: 1
Y right: -1
Alpha: 1.4354338174644021
Weights: [8.931e-01 5.059e-02 1.377e-04 7.427e-06 7.427e-06 2.615e-03 4.818e-02
 2.491e-03 2.730e-03 1.377e-04]
Y: [-1 -1  1  1 -1 -1  1  1 -1 -1]

===========> Round 5 : <===========
X: [[2 2 3 3 4 4 5 5 6 6]]
Threshold: 3.5
Y left: -1
Y right: 1
Alpha: 2.63792035905396
Weights: [9.436e-01 5.346e-02 7.437e-07 7.847e-06 4.012e-08 1.413e-05 2.603e-04
 2.632e-03 1.475e-05 7.437e-07]
Y: [ 1  1 -1  1  1  1 -1 -1  1  1]

===========> Round 6 : <===========
X: [[1 2 3 5 5 5 6 6 6 6]]
Threshold: 5.5
Y left: -1
Y right: 1
Alpha: 2.6128389446369225
Weights: [8.257e-02 8.700e-01 6.508e-08 1.277e-04 3.511e-09 1.237e-06 4.237e-03
 4.283e-02 2.400e-04 6.508e-08]
Y: [ 1  1 -1 -1 -1  1 -1 -1 -1  1]

===========> Round 7 : <===========
X: [[ 2  3  3  3  3  3  6  8 10 10]]
Threshold: 4.5
Y left: -1
Y right: -1
Alpha: 2.3919401489043093
Weights: [5.496e-02 5.791e-01 4.332e-08 8.500e-05 2.337e-09 8.230e-07 3.372e-01
 2.851e-02 1.598e-04 4.332e-08]
Y: [-1 -1 -1 -1 -1 -1 -1 -1 -1 -1]

===========> Round 8 : <===========
X: [[ 1  1  1  2  2  3  5  5  8 10]]
Threshold: 1.5
Y left: 1
Y right: -1
Alpha: 2.596858181607915
Weights: [8.272e-02 8.716e-01 3.619e-10 7.102e-07 1.953e-11 6.877e-09 2.817e-03
 4.291e-02 1.335e-06 3.619e-10]
Y: [-1 -1  1 -1 -1 -1 -1 -1 -1 -1]

===========> Round 9 : <===========
X: [[ 1  3  4  5  6  6  6  6  8 10]]
Threshold: 7.0
Y left: 1
Y right: -1
Alpha: 2.6915509043653616
Weights: [8.270e-02 8.713e-01 3.618e-10 7.100e-07 1.952e-11 6.875e-09 2.817e-03
 4.290e-02 2.905e-04 3.618e-10]
Y: [ 1 -1  1  1  1  1  1  1  1 -1]

===========> Round 10 : <===========
X: [[ 1  2  3  5  6  8  8  8  8 10]]
Threshold: 7.0
Y left: 1
Y right: -1
Alpha: 2.691696175836587
Weights: [7.537e-03 7.941e-02 7.181e-09 6.471e-08 1.779e-12 1.364e-07 5.590e-02
 8.514e-01 5.766e-03 3.298e-11]
Y: [ 1 -1  1  1  1  1  1  1  1 -1]

++++++++++ Result ++++++++++
Y True: [ 1 -1  1  1 -1  1 -1 -1  1 -1]
Sum: [ -0.203 -10.97    6.186  -2.558  -5.429  -0.203  -7.834  -1.97   -5.429
 -10.97 ]
Y Prediction: [-1. -1.  1. -1. -1. -1. -1. -1. -1. -1.]
Accuracy: 0.6
"""
