import matplotlib.pyplot as plt
import numpy as np
import util
from  math import inf
from p05b_lwr import LocallyWeightedLinearRegression
from p05b_lwr import myplot
import matplotlib.pyplot as plt

Tau = [6e-1, 5e-1, 2e-1, 1e-1, 5e-2,4e-2, 2e-2]

def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    x_test, y_test = util.load_dataset('data/ds5_test.csv', add_intercept = True)
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept = True)

    Lowest_mse = inf
    Best_tau = tau_values[0]
    for tau in tau_values:
        lwr = LocallyWeightedLinearRegression(tau)
        lwr.fit(x_train,y_train)
        pred = lwr.predict(x_valid)
        Best_tau = tau
        mse = np.mean((y_valid -pred))**2
        if mse < Lowest_mse:
            Lowest_mse = mse
        print(f'The tau is {tau}, the MSE is {mse}')
        myplot(x_valid, y_valid,pred  , f'tau = {tau}')
        plt.show()
    print(f'the best tau is {Best_tau}, the MSE is {Lowest_mse}')
    lwr = LocallyWeightedLinearRegression(Best_tau)
    lwr.fit(x_train,y_train)
    myplot(x_test, y_test, lwr.predict(x_test), 'Test set')
    plt.show()

if __name__ == '__main__':
    main(Tau, 'data/ds5_train.csv', 'data/ds5_valid.csv', test_path='data/ds5_test.csv', pred_path=None)

