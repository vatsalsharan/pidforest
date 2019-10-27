import numpy as np
import math
import matplotlib.pyplot as plt


# computes the summed squared error of locations i,i+1,...,j
def seg_var(i, j, v, vv):
    if j < i:
        return 0
    l = j - i + 1
    if i == 0:
        bx2 = 0
        bx = 0
    else:
        bx2 = vv[i - 1]
        bx = v[i - 1]
    x2 = (vv[j] - bx2) / l
    x = (v[j] - bx) / l
    # returns (E(x^2)-E^2(x)) * #items
    return (x2 - (x * x)) * l


def exact_buckets(A, k):
    v = np.cumsum(A)
    vv = np.cumsum(A ** 2)
    n = len(A)
    err = np.reshape(np.zeros(k * n), (n, k))
    borders = np.reshape(np.zeros(k * n), (n, k))

    # The base case of the dynamic program
    for i in range(n):
        err[i, 0] = seg_var(0, i, v, vv)
        borders[i, 0] = 0  # with one bucket the last bucket starts at zero.

    for p in range(1, k):
        for i in range(n):
            cur_err = seg_var(0, i, v, vv)
            for x in range(1, i + 1):  # x is a candidate for creating a new bucket [x....i]
                tmp = err[x - 1, p - 1] + seg_var(x, i, v, vv)
                if tmp <= cur_err:
                    cur_err = tmp
                    borders[i, p] = x
            err[i, p] = cur_err
    return err, borders
    # return OptHistMat, borders


def app_buckets(A, B, eps):
    a_values = {}
    b_values = {}
    for i in range(B):
        a_values[i] = [0]
        b_values[i] = []
    v = np.cumsum(A)
    vv = np.cumsum(A ** 2)
    n = len(A)
    apx_err = np.reshape(np.zeros(n * B), (n, B))
    borders = np.reshape(np.zeros(B * n), (n, B))
    for j in range(n):
        apx_err[j, 0] = seg_var(0, j, v, vv)
        a = a_values[0][-1]
        if (j >= 1) and (apx_err[j, 0] > (1 + eps) * apx_err[a, 0]):
            b_values[0].append(j - 1)
            a_values[0].append(j)
        for k in range(1, B):
            apx_err[j, k] = seg_var(0, j, v, vv)
            for i in b_values[k - 1]:
                tmp = apx_err[i, k - 1] + seg_var(i + 1, j, v, vv)
                if tmp < apx_err[j, k]:
                    apx_err[j, k] = tmp
                    borders[j, k] = i + 1
            a = a_values[k][-1]
            if k < B and (j >= 1) and (apx_err[j, k] > (1 + eps) * apx_err[a, k]):
                b_values[k].append(j - 1)
                a_values[k].append(j)
    return apx_err[-1], borders


# params:
# borders - a matrix computed by compute_hist
# k - number of buckets needed.
# returns:
# the indices of the k beginings of buckets.
def compute_buckets(borders, k):
    buckets = np.zeros(k)
    n, B = np.shape(borders)
    loc = n - 1
    for i in range(k):
        x = borders[loc, k - i - 1]
        buckets[k - i - 1] = x
        loc = int(x - 1)
    return buckets


def gen_mix(num, mu):
    k = len(mu)
    mixture = []
    for i in range(k):
        normal = np.random.randn(num) + mu[i]
        mixture = np.concatenate((mixture, normal))
    return mixture


def fast_buckets(arr, max_buckets):
    n_arr = len(arr)
    num_sum = np.cumsum(arr)
    sq_sum = np.cumsum(arr ** 2)
    err = {0: [sq_sum[j] - num_sum[j] ** 2 / (j + 1) for j in range(n_arr)]}
    bucket = {0: np.zeros(n_arr)}
    for k in range(1, max_buckets):
        err[k] = [
            np.min([(err[k - 1][i] + sq_sum[j] - sq_sum[i] - (num_sum[j] - num_sum[i]) ** 2 / (j - i)) if j > i else
                    err[k - 1][j] for i in range(j + 1)]) for j in range(n_arr)]
        bucket[k] = [np.argmin([(err[k - 1][i] + sq_sum[j] - sq_sum[i] - (num_sum[j] - num_sum[i]) ** 2 / (j - i))
                                if j > i else err[k - 1][j] for i in range(j + 1)]) for j in range(n_arr)]
    return err, bucket


def compress(arr, eps):
    min_val = np.min(arr[np.nonzero(arr)])
    max_val = np.max(arr[np.nonzero(arr)])
    k = int(math.log(max_val / min_val) / math.log(1 + eps))
    sp_vals = sorted({min_val * ((1 + eps) ** i) for i in range(k + 1)})
    sp_indices = sorted(set(np.searchsorted(arr, sp_vals)))
    compress_arr = {index: arr[index] for index in sp_indices}
    return compress_arr


def fast_approx(arr, max_buckets, eps):
    n_arr = len(arr)
    num_sum = np.cumsum(arr)
    sq_sum = np.cumsum(arr ** 2)
    err = np.array([sq_sum[j] - num_sum[j] ** 2 / (j + 1) for j in range(n_arr)])
    compress_err = [{} for _ in range(max_buckets)]
    for k in range(1, max_buckets):
        compress_err[k-1] = compress(err, eps)
        err = np.array([np.min([(val_i + sq_sum[j] - sq_sum[i] - (num_sum[j] - num_sum[i]) ** 2 / (j - i)) if j > i else
                                val_i for i, val_i in compress_err[k-1].items()]) for j in range(n_arr)])
    return err, compress_err
