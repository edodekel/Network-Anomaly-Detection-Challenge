import numpy as np


def ba_func(y_pred, y_test_classes_num, N):
    # calculate the balanced accuracy where we give the last class ("other") a larger weight.

    err_count, total_count = get_err_count(y_test_classes_num, y_pred, N)

    num_classes = 23

    L = len(err_count) - 1
    ba = 0
    c = -1
    for key in err_count:
        c += 1
        if c == len(err_count) - 1:
            ba += (1 - err_count[key] / total_count[key]) * (num_classes - L)  # we give the "other" class a larger weight
        else:
            ba += (1 - err_count[key] / total_count[key])

    ba = ba / num_classes

    return ba


def get_err_count(y_test_classes_num, y_pred, N):
    # get number of errors per class
    err_count = {}
    total_count = {}
    for n in range(N):
        ind = np.where(y_test_classes_num == n)
        total_count[n] = len(ind[0])
        err_count[n] = np.sum(y_pred[ind] != n)

    return err_count, total_count


def optimize_thresholds_D(X, y_test_classes_num):
    # we find the minimal factor to multiply the "other" class prediction, that lowers the amount of errors for that class to zero

    f_vec = np.logspace(-2, 6, num=101)
    y_pred_init = np.argmax(X, axis=1)
    N = X.shape[1] - 1
    ba_init = ba_func(y_pred_init, y_test_classes_num, N)
    ind = np.where(y_test_classes_num == N)
    for f in f_vec:

        X_temp = X.copy()
        X_temp[:, N] = X_temp[:, N] * f  # multiply the last column by f
        y_pred = np.argmax(X_temp, axis=1)

        errs = np.sum(y_pred[ind] != N)  # calculate errors for the last class
        print(f"errs={errs}")
        if errs == 0:  # break if we reached zero errors.
            break

    # sanity:
    ba_fin = ba_func(y_pred, y_test_classes_num, N)
    err_count, _ = get_err_count(y_test_classes_num, y_pred, N)
    print(f"err_count_fin={err_count}")
    print(f"ba_init={ba_init}")
    print(f"ba_fin={ba_fin}")

    return f
