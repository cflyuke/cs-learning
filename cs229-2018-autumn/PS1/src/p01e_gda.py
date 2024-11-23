import numpy as np
import matplotlib.pyplot as plt
import util
from p01b_logreg import LogisticRegression
from linear_model import LinearModel

ds1 = 'data/ds1_train.csv'
ds2 = 'data/ds1_valid.csv'

def main(train_path, eval_path, pred_path = None):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_valid, y_valid = util.load_dataset(eval_path, add_intercept=False)
    

    ## 比较在不同训练集下logisticRegression和gda训练出来的数据拟合度的优劣程度
    log = LogisticRegression()
    log.fit(util.add_intercept(x_train),y_train)
    print('The accuracy for the log: ', np.mean(log.predict(util.add_intercept(x_valid)) == y_valid))

    gda = GDA()
    gda.fit(x_train,y_train)
    print('The accuracy for the gda: ', np.mean(gda.predict(x_valid) == y_valid))

    my_plot(x_valid, y_valid,theta_1 = log.theta, legend_1 = 'log', theta_2 = gda.theta, legend_2 = 'gda', title = 'compare')

    ## 将训练集1进行正态化
    transfered_x_train = np.stack((x_train[:,0], np.log(x_train[:,1])), axis = 1)
    transfered_x_valid = np.stack((x_valid[:,0], np.log(x_valid[:,1])), axis = 1)
    logT = LogisticRegression()
    logT.fit(util.add_intercept(transfered_x_train), y_train)
    print("The accuracy for the log: ",np.mean(logT.predict(util.add_intercept(transfered_x_valid)) == y_valid ))

    gda = GDA()
    gda.fit(transfered_x_train, y_train)
    print('The accuracy for the gda: ', np.mean(gda.predict(transfered_x_valid) == y_valid))

    my_plot(transfered_x_valid, y_valid,theta_1 = logT.theta, legend_1 = 'log', theta_2 = gda.theta, legend_2 = 'gda', title = 'compare' )


def my_plot(x, y, theta_1, legend_1 = None, theta_2 = None, legend_2=None, title = None, correction=1.0 ):
    plt.figure(figsize=(10,6))
    plt.plot(x[y==True, -2], x[y==True, -1], 'bx', linewidth = 2)
    plt.plot(x[y==False, -2], x[y==False, -1], 'go', linewidth = 2)

    x_1 = np.arange(min(x[:, -2]), max(x[:, -2]), 0.01)
    x_2 = -( theta_1[0]/ theta_1[2] * correction + theta_1[1]/ theta_1[2] * x_1)
    plt.plot(x_1, x_2, c='red', label = legend_1, linewidth = 2)

    if theta_2 is not None:
        x_1 = np.arange(min(x[:, -2]),max(x[:, -2]), 0.01)
        x_2 = -(theta_2[0]/ theta_2[2] * correction + theta_2[1]/ theta_2[2] * x_1)
        plt.plot(x_1,x_2,c ='yellow', label = legend_2, linewidth = 2)
    
    plt.xlabel('x1')
    plt.ylabel('x2')
    if legend_1 is not None or legend_2 is not None:
        plt.legend(loc = "upper right")
    if title is not None:
        plt.suptitle(title, fontsize = 12)
    plt.show()


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        m, n = x.shape
        phi = np.sum(y)/m
        mu_0 = np.dot(x.T, 1-y) / (m - np.sum(y))
        mu_1 = np.dot(x.T, y) / np.sum(y)

        y = np.reshape(y, (m,1))
        mu_x = y * mu_1 + (1-y) * mu_0

        sigma = 1/m * np.dot((x - mu_x).T, (x - mu_x))                                                                                                                                               
        sigma_inv = np.linalg.inv(sigma)

        reside = -1/2 * mu_1 @ sigma_inv @ mu_1 + 1/2 *  mu_0 @ sigma_inv @ mu_0 - np.log((1-phi)/phi)
        theta_0 = np.dot(sigma_inv, mu_1 - mu_0)
        self.theta = np.insert(theta_0, 0, reside)



    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        return np.dot(util.add_intercept(x), self.theta) >= 0

if __name__ == "__main__":
    main(ds1,ds2)