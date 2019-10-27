import copy
import numpy as np
import matplotlib.pyplot as plt
from py.hg import Histogram


BUCKETS = 10
EPS = 0.1

class SubCube():
    
    def __init__(self, **kwargs):
        self.depth = kwargs['depth']
        self.max_depth = kwargs['max_depth']
        if self.depth == 0:
            self.points = kwargs['points']
            self.dim, self.num_pts = np.shape(self.points)
            self.get_box()
            self.id_string = [0]
            self.sp_axis = -1
            self.indices = range(self.num_pts)
        else:
            self.start = kwargs['start']
            self.end = kwargs['end']
            self.id_string = kwargs['id']
            self.sp_axis = kwargs['axis']
            self.filter_pts(kwargs['points'], kwargs['indices'])                
        self.density = self.compute_density()
        self.child = {}
        if (self.depth < self.max_depth) and (self.num_pts > 1):
            self.find_split()

    def get_box(self):
        self.start = np.zeros(self.dim)
        self.end = np.zeros(self.dim)
        self.val = []
        self.count = []
        for axis in range(self.dim):
            val, count = np.unique(np.array(self.points[axis]), return_counts=True)
            self.val.append(val) 
            self.count.append(count)
            if len(self.val[axis]) <= 1:
                self.start[axis] = self.val[axis][0]
                self.end[axis] = self.val[axis][0]
            else:
                self.start[axis] = (3*self.val[axis][0] - self.val[axis][1])/2
                self.end[axis] = (3*self.val[axis][-1] - self.val[axis][-2])/2

    def compute_density(self): 
        vol = 1
        for i in range(self.dim):
            vol = vol*(self.end[i] - self.start[i])
        return vol/self.num_pts

    def __str__(self):
        str_val = "Id: " + str(self.id_string) + ", Axis: " + str(self.sp_axis) + "\n"
        str_val += "Boundary: "
        for i in range(self.dim):
            str_val += " [" + str(self.start[i]) + ", " + str(self.end[i]) + "]"
            if i < self.dim -1:
                str_val += " x"
            else:
                str_val += "\n"
        str_val += "Points:\n " + str(np.transpose(self.points)) + "\n"
        str_val += "Indices: " + str(self.indices) + "\n"
        return str_val
    
    def print_cube(self):
        print_list = [self]
        while print_list:
            node = print_list.pop(0)
            print(str(node))
            print_list.extend(node.child.values())

    def filter_pts(self, pts, indices):
        pts_ind = []
        list_ind = []
        self.dim, num_pts = np.shape(pts)
        for i in range(num_pts):
            in_cube = True
            for j in range(self.dim):
                if (pts[j][i] < self.start[j]) or (pts[j][i] >= self.end[j]):
                    in_cube = False
            if in_cube:
                pts_ind.append(i)
                list_ind.append(indices[i])
        self.points = pts[:, pts_ind]
        self.indices = list_ind
        self.num_pts = len(list_ind)
        self.val = {}
        self.count = {}
        for axis in range(self.dim):
            self.val[axis], self.count[axis] = np.unique(np.array(self.points[axis]), return_counts=True)
    
    def get_gaps(self, axis):
        vals = self.val[axis]
        if len(vals) <= 1:
            gap = [0]
        else:
            gap = np.zeros(len(vals))
            gap[0] = (vals[0] + vals[1])/2 - self.start[axis]
            gap[-1] = self.end[axis] - (vals[-1] + vals[-2])/2
            for i in range(1, len(vals)-1):
                gap[i] = (vals[i+1] - vals[i-1])/2
        return gap
        
    def find_split(self):
        var_red = np.zeros(self.dim)
        buckets = {}
        for axis in range(self.dim):
            if len(self.val[axis]) > 1:
                gap = self.get_gaps(axis)
                hist = Histogram(gap/self.count[axis], self.count[axis], BUCKETS, EPS)
                _, var_red[axis], buckets[axis] = hist.best_split()
            else:
                var_red[axis] = 0
        if np.max(var_red) > 0:
            best_axis = np.argmax(var_red)
            split_vals = [(self.val[best_axis][i-1] + self.val[best_axis][i])/2 \
                          for i in buckets[best_axis]]
            for i in range(len(split_vals) + 1):
                new_start = np.array(self.start)
                new_end = np.array(self.end)
                if i == 0:
                    new_end[best_axis] = split_vals[0]
                elif i == len(split_vals):
                    new_start[best_axis] = split_vals[-1]
                else:
                    new_start[best_axis] = split_vals[i-1]
                    new_end[best_axis] = split_vals[i]
                new_id = copy.deepcopy(self.id_string)
                new_id.append(i)
                kwargs = {'depth':self.depth + 1, 'max_depth':self.max_depth}
                kwargs.update({'start':new_start, 'end':new_end, 'axis':best_axis})
                kwargs.update({'points':self.points, 'indices':self.indices, 'id':new_id}) 
                self.child[i] = SubCube(**kwargs)
            
class Partition():

    def __init__(self, cube):
        not_leaves = [cube]
        leaves = []
        density = []
        while not_leaves:
            this_cube = not_leaves.pop(0)
            for next_cube in this_cube.child.values():
                if next_cube.child:
                    not_leaves.append(next_cube)
                else:
                    leaves.append(next_cube)
                    density.append(next_cube.density)
        self.num_leaves = len(leaves)
        sort_density = np.flip(np.argsort(density), axis=0)
        self.density = np.flip(np.sort(density), axis=0)
        self.leaves = []
        self.indices = []
        self.num_pts = []
        for i in sort_density:
            self.leaves.append(leaves[i])
            self.indices.append(leaves[i].indices)
            self.num_pts.append(leaves[i].num_pts)
        plt.plot(self.density)
        plt.show()
        plt.plot(self.num_pts)
        plt.show()

    def get_top(self, k, to_print=False):
        tot_pts = 0
        count = 0
        top_list = []
        while tot_pts < k:
            top_list.append(self.indices[count])
            tot_pts += self.num_pts[count]
            count += 1
        if to_print:
            str_val = ""
            for i in range(count):
                str_val += str(self.leaves[i]) + "Density: "  + str(self.density[i]) + "\n\n"
            print(str_val)
        return top_list, count

    def __str__(self):
        str_val = ""
        for i in range(self.num_leaves):
            str_val += str(self.leaves[i]) + "Density: "  + str(self.density[i]) + "\n\n"
        return str_val
