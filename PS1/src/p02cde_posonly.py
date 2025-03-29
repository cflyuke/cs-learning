import numpy as np
import util
import matplotlib.pyplot as plt
from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    _, t_train = util.load_dataset(train_path, label_col = 't')
    x_test, y_test = util.load_dataset(test_path, add_intercept = True)
    _, t_test = util.load_dataset(test_path, label_col = 't')
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept = True)
    _, t_valid = util.load_dataset(valid_path, label_col = 't')

    # *** START CODE HERE ***
    # Part (c): Train and test on true labels
    # Make sure to save outputs to pred_path_c
    
    log_c = LogisticRegression()
    log_c.fit(x_train, t_train)
    print("The accuracy on test set for e is : ", np.mean(log_c.predict(x_test) == t_test))
    record(pred_path_c, log_c.predict(x_test))
    # util.plot(x_train, t_train, log_c.theta)
    # print("The accuracy on train set is : ", np.mean(log_c.predict(x_train) == t_train))
    # plt.show()
    # util.plot(x_test, t_test, log_c.theta)
    # plt.show()




    # Part (d): Train on y-labels and test on true labels
    # Make sure to save outputs to pred_path_d

    log_d = LogisticRegression()
    log_d.fit(x_train, y_train)
    print("The accuracy on test set for d is : ", np.mean(log_d.predict(x_test) == t_test))
    record(pred_path_d, log_d.predict(x_test))
    # util.plot(x_test, t_test, log_d.theta)
    # plt.show()




    # Part (e): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to pred_path_e
    log_e = LogisticRegression()
    log_e.fit(x_valid, y_valid)

    x_valid_1 = x_valid[y_valid == 1,:]
    alpha = np.mean(1 / (1 + np.exp(-np.dot(x_valid_1, log_e.theta))) )

    log_e.theta = log_e.theta + np.log(2/alpha -1) *np.array([1,0,0])
    pred_result = log_e.predict(x_valid) == t_valid
    print("The accuracy for the valid set is : ", np.mean(pred_result))
    record(pred_path_e, pred_result)
    # util.plot(x_test, t_test, log_e.theta)
    # plt.show()
    
    my_plot(x_test, t_test, theta_1 = log_c.theta, theta_2 = log_d.theta, theta_3 = log_e.theta, legend_1 = 'c', legend_2='d', legend_3='e')


    # *** END CODER HERE
def my_plot(x, t, theta_1 = None, theta_2 = None, theta_3 = None, legend_1 = None , legend_2 = None, legend_3 = None, correction = 1.0):
    plt.figure(figsize=(10,6))
    plt.plot(x[t == 1, -2], x[ t == 1, -1], 'bx', linewidth = 2)
    plt.plot(x[t == 0, -2], x[ t == 0, -1], 'go', linewidth = 2)
    x_1 = np.arange(min(x[:, -2]), max(x[:, -2]), 0.01)
    if theta_1 is not None:
        x_2 = -(theta_1[0]/theta_1[2] * correction + theta_1[1]/ theta_1[2] *x_1)
        plt.plot(x_1, x_2, c = 'blue', label = legend_1, linewidth = 2)
    if theta_2 is not None:
        x_2 = -(theta_2[0]/theta_2[2] * correction + theta_2[1]/ theta_2[2] *x_1)
        plt.plot(x_1, x_2, c = 'red', label = legend_2, linewidth = 2)
    if theta_3 is not None:
        x_2 = -(theta_3[0]/theta_3[2] * correction + theta_3[1]/ theta_3[2] *x_1)
        plt.plot(x_1, x_2, c = 'yellow', label = legend_3, linewidth = 2)
    margin1 = (max(x[:, -2]) - min(x[:, -2]))*0.2
    margin2 = (max(x[:, -1]) - min(x[:, -1]))*0.2
    plt.xlim(x[:, -2].min()-margin1, x[:, -2].max()+margin1)
    plt.ylim(x[:, -1].min()-margin2, x[:, -1].max()+margin2)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend(loc = 'upper right')
    plt.suptitle('Compare')
    plt.show()


def record(path, pred_result):
    with open(path, 'w') as f:
        for pred in pred_result:
            f.write(f"{pred}\n")


if __name__ == "__main__":
    main('data/ds3_train.csv', 'data/ds3_valid.csv', 'data/ds3_test.csv', 'src/results/pred_X.txt')