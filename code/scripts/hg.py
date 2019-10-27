import numpy as np
import math


def compress(arr, eps):
    min_val = np.min(arr[np.nonzero(arr)])
    max_val = np.max(arr[np.nonzero(arr)])
    k = int(math.log(max_val / min_val) / math.log(1 + eps))
    sp_vals = sorted({min_val * ((1 + eps) ** i) for i in range(k + 1)})
    sp_indices = sorted(set(np.searchsorted(arr, sp_vals)))
    compress_arr = {index: arr[index] for index in sp_indices}
    return compress_arr

class Histogram:

    def __init__(self, val, count, max_buckets, eps):
        self.num = len(val)
        self.max_buckets = max_buckets
        self.val = val
        self.count = count
        self.eps = eps
        self.err, self.b_values = approx_buckets(val, count, max_buckets, eps)

    def __str__(self):
        str_val = ""
        for i in range(self.max_buckets):
            str_val += "Level " + str(i) + ":"
            for b_val in self.b_values[i]:
                a_val, err_a, _, _, _ = self.b_values[i][b_val]
                str_val += " (" + str(a_val) + ", " + str(b_val) + "): " + str(err_a) + ", "
            str_val += "\n"
        return str_val

    def test(self):
        print(str(self))
        for i in range(1, self.max_buckets):
            print(self.compute_buckets(i))
        print("Best Buckets: ")
        print(self.best_split())

    def best_split(self):
        if self.err[0] == 0:
            return 0, 0, []
        err_red = [(self.err[0] - self.err[i]) for i in range(1, self.max_buckets)]
        var_red = np.max(err_red) / self.err[0]
        if var_red < 0:
            print("error: var_red is", var_red)
            var_red = 0
        opt = np.argmax(err_red) + 2
        buckets = self.compute_buckets(opt)
        return opt, var_red, buckets[1:]

    def compute_buckets(self, num_buckets):
        buckets = []
        end = self.num - 1
        k = num_buckets - 1
        while end >= 0:
            start = int(self.b_values[k][end][0])
            if start <= end:
                buckets.append(start)
            end = start - 1
            k -= 1
        return np.flip(buckets, axis=0)


def two_split(val, count):
    p_count = np.cumsum(count)
    s_count = p_count[-1] - p_count
    p_sum = np.cumsum(val*count)
    s_sum = p_sum[-1] - p_sum
    scores = (p_sum**2)[:-1]/p_count[:-1] + (s_sum**2)[:-1]/s_count[:-1]
    return scores


def approx_buckets(arr, count, max_buckets, eps):
    """params:
    vals: the array of values
    counts: the array of counts
    max_buckets: the number of buckets
    eps: an approximation parameter
    returns:
     1) an array cur_err[k], which gives the error of the best histogram with k buckets.
     2) a dictionary b_values.
    b_values stores a collection of intervals for each level k where 0 <= k < B. It is indexed by
    the level k and the endpoint b of an interval (a,b) at level k.
    The value is a 4 tuple:
    1) a: start point of the interval
    2) ApxError(b,k) for that point.
    3) sum until b
    4) sum of squares until b
    5) total count until b"""
    err_a = np.zeros(max_buckets) - 1
    cur_err = np.zeros(max_buckets)
    b_values = [{} for _ in range(max_buckets)]
    cur_sum = 0
    cur_sq = 0
    cur_pts = 0
    for j in range(len(arr)):
        cur_sum += arr[j] * count[j]
        cur_sq += (arr[j] ** 2) * count[j]
        cur_pts += count[j]
        cur_err[0] = cur_sq - cur_sum**2/cur_pts
        if cur_err[0] > (1 + eps) * err_a[0]:
            err_a[0] = cur_err[0]
        else:
            del b_values[0][j - 1]
        b_values[0][j] = (0, cur_err[0], cur_sum, cur_sq, cur_pts)
        for k in range(1, max_buckets):
            cur_err[k] = cur_err[k - 1]
            a_val = j + 1
            for b_val in b_values[k - 1].keys():
                if b_val < j:
                    _, b_err, b_sum, b_sq, b_pts = b_values[k - 1][b_val]
                    tmp_error = b_err + cur_sq - b_sq - (cur_sum - b_sum) ** 2 / (cur_pts - b_pts)
                    if tmp_error < cur_err[k]:
                        cur_err[k] = tmp_error
                        a_val = b_val + 1
            b_values[k][j] = (a_val, cur_err[k], cur_sum, cur_sq, cur_pts)
            if cur_err[k] > (1 + eps) * err_a[k]:
                err_a[k] = cur_err[k]
            else:
                del b_values[k][j - 1]
    return cur_err, b_values
