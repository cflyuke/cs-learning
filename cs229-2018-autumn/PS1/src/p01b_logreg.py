import numpy as np
import util
import matplotlib.pyplot as plt
from linear_model import LinearModel

ds1_training_set_path = 'data/ds1_train.csv'
ds1_valid_set_path = 'data/ds1_valid.csv'

def main(train_path, eval_path, pred_path = None):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept = True)
    x_valid, y_valid = util.load_dataset(eval_path, add_intercept = True)
    log_reg = LogisticRegression()
    log_reg.fit(x_train, y_train)
    util.plot(x_train, y_train, theta = log_reg.theta)
    print('Theta is: ', log_reg.theta)
    print('The accuracy on the training set is: ', np.mean(log_reg.predict(x_train) == y_train))
    plt.show()
    util.plot(x_valid, y_valid,theta = log_reg.theta)
    print('The accuracy on the validation set is: ', np.mean(log_reg.predict(x_valid) == y_valid))
    plt.show()

class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        def h(theta, x):
            return 1/ (1 + np.exp(-np.dot(x, theta)))
        
        def gradient(theta, x, y):
            m , _ = x.shape
            return -1/ m * np.dot(x.T, (y - h(theta, x))) 
        
        def hessian(theta, x):
            m , _ = x.shape
            htheta_x = h(theta, x).reshape(-1,1)
            return 1/m * (x.T @ (htheta_x * (1 - htheta_x) * x))
        
        def next_theta(theta, x, y):
            return theta - np.dot(np.linalg.inv(hessian(theta, x)), gradient(theta, x, y))
        
        m, n = x.shape
        if self.theta is None:
            self.theta = np.zeros(n)

        old_theta = self.theta
        new_theta = next_theta(self.theta, x, y)
        while np.linalg.norm(old_theta - new_theta) >= self.eps:
             old_theta = new_theta
             new_theta = next_theta(old_theta, x, y)
        self.theta = new_theta


    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        return np.dot(x, self.theta) >= 0

if __name__ == '__main__':
   main(ds1_training_set_path,ds1_valid_set_path)