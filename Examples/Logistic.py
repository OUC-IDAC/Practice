import numpy as np
import matplotlib.pyplot as plt


def data_set():
    np.random.seed(4)
    num_observations = 500
    x1 = np.random.multivariate_normal([2, 5], [[1, 0.6], [0.6, 1]], num_observations)
    x2 = np.random.multivariate_normal([1, 1], [[1, 0.6], [0.6, 1]], num_observations)
    permutation = np.random.permutation(2*num_observations)
    x = np.vstack((x1, x2)).astype(np.float32)[permutation]
    y = np.hstack((np.zeros(num_observations), np.ones(num_observations)))[permutation]
    print(y)
    x_train = x[:800]
    y_train = y[:800]
    x_test = x[800:]
    y_test = y[800:]
    return x_train, y_train, x_test, y_test


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def gradient(features, target, predictions):
    return -np.dot(features.T, target - predictions)


def log_likelihood(features, target, weights):
    scores = np.dot(features, weights)
    ll = np.sum(target*scores - np.log(1 + np.exp(scores)))
    return ll


def logistic_regression(features, target, num_steps, learning_rate):
    intercept = np.ones((features.shape[0], 1))
    features = np.hstack((intercept, features))
    num = 1000

    weights = np.zeros(features.shape[1])
    print(features.shape[0], features.shape[1])

    for steps in range(num_steps):
        predictions = sigmoid(np.dot(features, weights))
        weights -= learning_rate * gradient(features, target, predictions)

        if steps % num == 0:
            print('Steps: {} / {}'.format(steps / num, num_steps / num))
            print('Log Likelihood: ', log_likelihood(features, target, weights))
    return weights


def test():
    x_train, y_train, x_test, y_test = data_set()

    plt.figure(figsize=(10, 10))
    plt.scatter(x_train[:, 0], x_train[:, 1],
                c=y_train)
    plt.show()

    weights = logistic_regression(x_train, y_train,
                                  num_steps=30000, learning_rate=5e-5)

    train_data_with_intercept = np.hstack((np.ones((x_train.shape[0], 1)), x_train))
    test_data_with_intercept = np.hstack((np.ones((x_test.shape[0], 1)), x_test))

    train_preds = np.round(sigmoid(np.dot(train_data_with_intercept, weights)))
    test_preds = np.round(sigmoid(np.dot(test_data_with_intercept, weights)))
    print('Train Accuracy: {0}'.format((train_preds == y_train).sum().astype(float) / len(train_preds)))
    print('Test Accuracy: {0}'.format((test_preds == y_test).sum().astype(float) / len(test_preds)))


if __name__ == '__main__':
    test()