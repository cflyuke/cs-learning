import matplotlib.pyplot as plt
import numpy as np
import util
from linear_model import LinearModel

#个人的一点感受就是：这个LWR算法就是对于线性逼近的一点优化，通过保留原有数据来对给定的数据进行更准确的预测，即加大距离给出的x更近的样本权重，从而在局部的预测精确度达更高的水平。
#tau又叫Bandwidth, 可以想象的是tau越小一般局部的预测精确度更高。




def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept = True)
    x_test, y_test = util.load_dataset('data/ds5_test.csv', add_intercept = True)
    x_valid, y_valid = util.load_dataset(eval_path, add_intercept = True)

    lwr = LocallyWeightedLinearRegression(tau)
    lwr.fit(x_train,y_train)
    myplot(x_train, y_train, lwr.predict(x_train), 'Train set' )
    plt.show()
    myplot(x_valid, y_valid, lwr.predict(x_valid), 'Validate set')
    plt.show()

def myplot(x, y, pred, title):
    plt.figure(figsize=(10,6))
    plt.plot(x[:,-1], y, 'bx', label = 'label')
    plt.plot(x[:,-1], pred, 'ro', label = 'prediction')
    plt.suptitle(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend('upper left')



class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        self.x = x
        self.y = y
        

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        m , n = x.shape
        weighted = np.apply_along_axis(np.diag, axis = 1, arr = np.exp(- np.linalg.norm(self.x - np.reshape(x, (m, -1, n)), ord = 2, axis = 2)**2/(2*self.tau**2)))
        theta = np.linalg.inv(self.x.T @weighted @self.x) @ self.x.T @ weighted @ self.y
        return np.sum(x*theta, axis = 1)

if __name__ == '__main__':
    main(0.5, 'data/ds5_train.csv','data/ds5_valid.csv' )