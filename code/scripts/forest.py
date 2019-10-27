import copy
from typing import List
import numpy as np
from numpy.core import ndarray
from scripts.hg import Histogram
import matplotlib.pyplot as plt
from math import log

to_print = False


class Forest:
    """This creates a forest of trees, given the following list of parameters:
    1) n_trees: the number of trees
    2) max_depth: the depth of the trees
    3) max_samples: number of samples per tree
    4) max_buckets: maximum number of buckets used by the histogram
    5) epsilon: accuracy of the histogram"""
    dim = ...  # type: int
    size = ...  # type: int
    points = ...  # type: ndarray
    start = ...  # type: ndarray
    end = ...  # type: ndarray

    def __init__(self, **kwargs):
        self.n_trees = kwargs['n_trees']
        self.max_depth = kwargs['max_depth']
        self.max_samples = kwargs['max_samples']
        self.max_buckets = kwargs['max_buckets']
        self.epsilon = kwargs['epsilon']
        self.sample_axis = kwargs['sample_axis']
        self.threshold = kwargs['threshold']
        self.tree = []
        # np.random.seed(seed = 17)
        self.n_leaves = np.zeros(self.n_trees)

    def fit(self, pts):
        self.points = pts
        self.dim, self.size = np.shape(self.points)
        if int(self.sample_axis*self.dim) == 0:
            print("sample_axis is too low")
            return
        self.start = np.zeros(self.dim)
        self.end = np.zeros(self.dim)
        for axis in range(self.dim):
            val = np.unique(np.array(self.points[axis]))
            if len(val) <= 1:
                print("No entropy in dimension :", axis)
                return
            self.start[axis] = (3 * val[0] - val[1]) / 2
            self.end[axis] = (3 * val[-1] - val[-2]) / 2
        k_args = {'depth': 0, 'forest': self}
        max_sample_size = np.min((self.size, self.max_depth*200))
        sample = np.random.choice(self.size, max_sample_size, replace=False)
        for i in range(self.n_trees):
            k_args['indices'] = np.random.choice(self.size, self.max_samples, replace=False)
            root_node = Node(**k_args)
            root_node.compute_density(sample)
            self.tree.append(root_node)
            self.n_leaves[i] = root_node.compute_leaf_num()

    def plt_scores(self, pts):
        _, n_pts = np.shape(pts)
        n_show = int(2 * self.n_trees / 3)
        scores = np.zeros((self.n_trees, n_pts))  # need to do this streaming
        indices = [i for i in range(n_pts)]
        for i in range(self.n_trees):
            self.tree[i].compute_split(pts, indices, scores[i])
        for i in range(n_pts):
            plt.plot(np.sort(scores[:, i])[:n_show])
            plt.show()

    def predict(self, pts, err=0.1, pct=50):
        _, n_pts = np.shape(pts)
        scores = np.zeros((self.n_trees, n_pts))  # need to do this streaming
        indices = [i for i in range(n_pts)]
        for i in range(self.n_trees):
            self.tree[i].compute_split(pts, indices, scores[i])
        n_err = int(err * n_pts)
        min_score = np.percentile(scores, pct, axis=0)
        top_indices = np.argsort(min_score)[:n_err]
        anom_pts = {}
        anom_scores = {}
        anom_pct = {}
        for i in range(n_err):
            anom_pts[top_indices[i]] = pts[:, top_indices[i]]
            anom_scores[top_indices[i]] = scores[:, top_indices[i]]
            anom_pct[top_indices[i]] = min_score[top_indices[i]]
        return top_indices, anom_pts, anom_scores, anom_pct, min_score
    
    def predict_depth(self, pts, err=0.1, pct=50):
        _, n_pts = np.shape(pts)
        scores = np.zeros((self.n_trees, n_pts))  # need to do this streaming
        indices = [i for i in range(n_pts)]
        for i in range(self.n_trees):
            self.tree[i].compute_depth(pts, indices, scores[i])
        n_err = int(err * n_pts)
        min_score = np.percentile(scores, pct, axis=0)
        top_indices = np.argsort(min_score)[:n_err]
        anom_pts = {}
        anom_scores = {}
        anom_pct = {}
        for i in range(n_err):
            anom_pts[top_indices[i]] = pts[:, top_indices[i]]
            anom_scores[top_indices[i]] = scores[:, top_indices[i]]
            anom_pct[top_indices[i]] = min_score[top_indices[i]]
        return top_indices, anom_pts, anom_scores, anom_pct, min_score
    
    def find_min_cube(self, pts, n_err=1, pct1=50, pct2=50):
        _, n_pts = np.shape(pts)
        scores = np.zeros((self.n_trees, n_pts))  # need to do this streaming
        indices = [i for i in range(n_pts)]
        for i in range(self.n_trees):
            self.tree[i].compute_split(pts, indices, scores[i])
        min_score = np.percentile(scores, pct1, axis=0)
        top_indices = np.argsort(min_score)[:n_err]
        sort_trees = np.argsort(scores, axis=0)
        min_tree = sort_trees[int(self.n_trees*pct2/100.0)]
        boxes = []
        for i in range(0,n_err):
            print('----')
            best_tree = min_tree[top_indices[i]]
            boxes.append( self.tree[best_tree].compute_box(pts[:,top_indices[i]]) )
            # boxes = self.tree[best_tree].compute_box(pts[:,top_indices[i]])
        return top_indices, boxes


class PointSet:
    def __init__(self, node, indices):
        self.node = node
        self.indices = indices
        self.val = []
        self.count = []
        self.gap = []
        for axis in range(self.node.forest.dim):
            val, count = np.unique(np.array(self.node.forest.points[axis, self.indices]), return_counts=True)
            self.val.append(val)
            self.count.append(count)
            if len(val) <= 1:
                gap = [0]
            else:
                gap = np.zeros(len(val))
                gap[0] = (val[0] + val[1]) / 2 - self.node.cube.start[axis]
                gap[-1] = self.node.cube.end[axis] - (val[-1] + val[-2]) / 2
                for i in range(1, len(val) - 1):
                    gap[i] = (val[i + 1] - val[i - 1]) / 2
            self.gap.append(gap)


class Cube:
    def __init__(self, node, start, end):
        assert isinstance(node, Node)
        self.node = node
        self.child = []
        self.start = start
        self.end = end
        self.dim = len(start)
        self.split_axis = -1
        self.split_vals = []
        self.vol = 0
        for i in range(self.dim):
            #print(self.end[i] - self.start[i])
            self.vol += log(self.end[i] - self.start[i])
        #print('testing')

    def filter_indices(self, indices):
        in_lb = self.node.forest.points[:, indices] >= self.start.reshape(self.dim, 1)
        in_ub = self.node.forest.points[:, indices] < self.end.reshape(self.dim, 1)
        return [indices[i] for i in range(len(indices)) if in_lb[:, i].all() and in_ub[:, i].all()]

    def split_indices(self, pts, indices):
        n_child = len(self.child)
        if n_child == 0:
            return indices
        n_arr = len(indices)
        if n_arr == 0:
            return [[] for _ in range(n_child)]
        s_arr = pts[self.split_axis]
        s_start = self.start[self.split_axis]
        s_end = self.end[self.split_axis]
        index_split = [[] for _ in range(n_child)]
        index_split[0] = [ind for ind in indices if ((s_arr[ind] >= s_start) and (s_arr[ind] < self.split_vals[0]))]
        index_split[-1] = [ind for ind in indices if ((s_arr[ind] >= self.split_vals[-1]) and (s_arr[ind] < s_end))]
        for k in range(1, n_child - 1):
            index_split[k] = [ind for ind in indices if (s_arr[ind] >= self.split_vals[k - 1]) and
                              (s_arr[ind] < self.split_vals[k])]
        return index_split

class Node:

    def __init__(self, depth, forest, **kwargs):
        self.depth = depth
        self.forest = forest
        if self.depth == 0:
            self.id_string = [0]
            self.cube = Cube(self, self.forest.start, self.forest.end)
            self.point_set = PointSet(self, kwargs['indices'])
        else:
            self.id_string = kwargs['id']
            self.cube = Cube(self, kwargs['start'], kwargs['end'])
            self.point_set = PointSet(self, self.cube.filter_indices(kwargs['indices']))
        self.density = -1
        self.child = []
        if (self.depth < self.forest.max_depth) and (len(self.point_set.indices) > 1):
            self.find_split()

    def find_split(self):
        imp_axis = [axis for axis in range(self.cube.dim) if len(self.point_set.val[axis]) > 1]
        if not imp_axis:
            return
        max_axes = min(len(imp_axis), int(self.forest.sample_axis * self.cube.dim))
        s_axes = np.random.choice(imp_axis, max_axes, replace=False)
        buckets = {}
        var_red = {}
        for axis in s_axes:
            hist = Histogram(self.point_set.gap[axis] / self.point_set.count[axis], self.point_set.count[axis],
                             self.forest.max_buckets, self.forest.epsilon)
            _, var_red[axis], buckets[axis] = hist.best_split()
        if np.max(list(var_red.values())) <= self.forest.threshold:
            return
        split_axis = np.random.choice(s_axes, p=list(var_red.values()) / np.sum(list(var_red.values())))
        self.cube.split_axis = split_axis
        self.cube.split_vals = [(self.point_set.val[split_axis][i - 1] + self.point_set.val[split_axis][i]) / 2 for i in
                                buckets[split_axis]]
        for i in range(len(self.cube.split_vals) + 1):
            new_start = np.array(self.cube.start)
            new_end = np.array(self.cube.end)
            if 0 < i < len(self.cube.split_vals):
                new_start[split_axis] = self.cube.split_vals[i - 1]
                new_end[split_axis] = self.cube.split_vals[i]
            elif i == 0:
                new_end[split_axis] = self.cube.split_vals[0]
            else:  # i == len(self.cube.split_vals)
                new_start[split_axis] = self.cube.split_vals[-1]
            new_id = copy.deepcopy(self.id_string)
            new_id.append(i)
            kwargs = {'start': new_start, 'end': new_end}
            kwargs.update({'indices': self.point_set.indices, 'id': new_id})
            child_node = Node(self.depth + 1, self.forest, **kwargs)
            self.child.append(child_node)
            self.cube.child.append(child_node.cube)

    def compute_density(self, indices):
        num = len(indices)
        if num == 0:
            self.density = 0
            self.child = []
            self.cube.child = []
            self.cube.split_axis = -1
            return
        self.density = log(num) - self.cube.vol
        if self.child:
            index_split = self.cube.split_indices(self.forest.points, indices)
            for i in range(len(self.child)):
                self.child[i].compute_density(index_split[i])

    def compute_leaf_num(self):
        if self.child:
            leaves = 0
            for i in range(len(self.child)):
                leaves += self.child[i].compute_leaf_num()
            return leaves
        else:
            return 1

    def compute_split(self, pts, indices, scores):
        if self.child:
            index_split = self.cube.split_indices(pts, indices)
            for i in range(len(self.child)):
                if len(index_split[i]) > 0:
                    self.child[i].compute_split(pts, index_split[i], scores)
        else:
            scores[indices] = self.density
            
    def compute_box(self, pts):
        if self.child:
            n_child = len(self.child)
            s_arr = pts[self.cube.split_axis]
            print('splitting on axis '+str(self.cube.split_axis)) 
            s_start = self.cube.start[self.cube.split_axis]
            s_end = self.cube.end[self.cube.split_axis]
            if ((s_arr >= s_start) and (s_arr < self.cube.split_vals[0])):
                #print('splitting on axis '+str(self.cube.split_axis)) 
                return(self.child[0].compute_box(pts))
            elif ((s_arr >= self.cube.split_vals[-1]) and (s_arr < s_end)):
                #print('splitting on axis '+str(self.cube.split_axis)) 
                return(self.child[-1].compute_box(pts))
            else:
                for k in range(1, n_child - 1):
                    if (s_arr >= self.cube.split_vals[k - 1]) and (s_arr < self.cube.split_vals[k]):
                        #print(str(k)+'splitting on axis '+str(self.cube.split_axis)+'\n') 
                        return(self.child[k].compute_box(pts))
        else:
            return self.cube
        
    def compute_depth(self, pts, indices, scores):
        if self.child:
            index_split = self.cube.split_indices(pts, indices)
            for i in range(len(self.child)):
                if len(index_split[i]) > 0:
                    self.child[i].compute_depth(pts, index_split[i], scores)
        else:
            scores[indices] = self.depth    

    def __str__(self):
        str_val = "Id: " + str(self.id_string) + "\n"
        str_val += "Boundary: "
        for i in range(self.cube.dim):
            str_val += " [" + str(self.cube.start[i]) + ", " + str(self.cube.end[i]) + "]"
            if i < self.cube.dim - 1:
                str_val += " x"
            else:
                str_val += "\n"
        str_val += "Points:\n " + str(np.transpose(self.point_set.points)) + "\n"
        str_val += "Indices: " + str(self.point_set.indices) + "\n"
        return str_val

    def print_node(self):
        print_list = [self]
        while print_list:
            node = print_list.pop(0)
            print(str(node))
            print_list.extend(node.child)
