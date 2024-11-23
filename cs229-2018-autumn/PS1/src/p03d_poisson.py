import numpy as np
import util
import matplotlib.pyplot as plt
from linear_model import LinearModel


def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept = True)
    x_valid, y_valid = util.load_dataset(eval_path, add_intercept = True)
    pos = PoissonRegression(step_size=lr)
    pos.fit(x_train, y_train)
    my_plot(y_valid, pos.predict(x_valid), legend1 = 'label', legend2 = 'pred', title = 'Poisson Pred')
    plt.show()
    
def my_plot(y, y_pred, legend1 = None, legend2 = None, title = None):
    try:
        plt.plot(y, 'go', label = legend1 )
        plt.plot(y_pred, 'rx', label = legend2)
    except ImportError :
        print("请输入标签!")
    plt.legend(loc = 'upper left')
    plt.suptitle(title, fontsize = 12)



class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def h(self, theta, x):
        return np.exp(np.dot(x,theta))

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        
        
        def next_step(theta):
            return self.step_size / m * np.dot(x.T, y - self.h(theta, x)) 

        
        m , n = x.shape
        if (self.theta == None):
            theta = np.random.randn(n)*0.01
        else :
            theta = self.theta

        step = next_step(theta)
        while np.linalg.norm(step, 1) >= self.eps:
            theta = theta + step
            step = next_step(theta)
        self.theta = theta


    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        return self.h(self.theta, x)
if __name__ == '__main__':
    main(2e-7, 'data/ds4_train.csv', 'data/ds4_valid.csv', pred_path=None)