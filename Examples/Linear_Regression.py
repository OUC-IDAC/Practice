import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_salary():
    path = 'D:/智能分析社团/Practice/Chapter 3/Salary_Data.csv'
    pd_data = pd.read_csv(path, ',', header=1)
    year = pd_data.T.values[0]
    salary = pd_data.T.values[1]
    return year, salary


def square_loss(w, b, x, y):
    loss = 0
    for i in range(len(x)):
        loss += pow(w*x[i] + b - y[i], 2)
    return loss


def closed_solve(x, y):
    average_x = np.average(x)
    size = len(x)
    w_closed = sum([(y[i] * (x[i] - average_x)) for i in range(size)]) / (sum([pow(x[i], 2) for i in range(size)]) - pow(sum(x), 2) / size)
    b_closed = sum([(y[i] - w_closed*x[i]) for i in range(size)]) / size
    return w_closed, b_closed


def gradient_solve(x, y, learning_rate, num_steps):
    w = 0
    b = 0
    size = len(x)
    for i in range(num_steps):
        w -= learning_rate * 2 * (w * sum([pow(x[i], 2) for i in range(size)]) - sum([((y[i] - b) * x[i]) for i in range(size)]))
        b -= learning_rate * 2 * (b * size - sum([(y[i] - w * x[i]) for i in range(size)]))
    return w, b


def main():
    x, y = load_salary()
    w_closed, b_closed = closed_solve(x, y)
    w_gradient, b_gradient = gradient_solve(x, y, 0.0003, 50000)
    print('closed solve: \nw = ', w_closed, '\nb = ', b_closed, '\nSquare Loss = ',
          square_loss(w_closed, b_closed, x, y))
    print('gradient solve: \nw = ', w_gradient, '\nb = ', b_gradient, '\nSquare Loss = ',
          square_loss(w_gradient, b_gradient, x, y))

    y_predict_closed = w_closed * x + b_closed
    y_predict_gradient = w_gradient * x + b_gradient
    plt.plot(x, y, 'or')
    plt.plot(x, y_predict_closed, 'b--', linewidth=3)
    plt.plot(x, y_predict_gradient, 'g-.', linewidth=5)
    plt.show()


main()
