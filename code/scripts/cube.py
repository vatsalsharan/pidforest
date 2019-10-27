import numpy as np
from py.forest import Node

class Cube():

    def __init__(self, node):
        assert isinstance(node, Node)
        self.start = node.start
        self.end = node.end
        self.dim = node.dim
        self.id_string = node.id_string
        self.split_axis = node.split_axis
        self.split_vals = node.split_vals
        self.vol = 1
        for i in range(len(self.start)):
            self.vol = self.vol*(self.end[i] - self.start[i])
        self.frac = 0
        self.child = [Cube(child_node) for child_node in node.child]

    def est_density(self, pts, total):
        self.frac = len(pts)/total
        if self.split_axis != -1:
            split_pts = self.split_points(pts)
            for i in range(len(self.child)):
                self.child[i].est_density(split_pts[i], total)

    def split_points(self, pts):
        _, num_pts = np.shape(pts)
        indices = [[] for _ in range(len(self.split_vals) + 1)]
        list_vals = [self.start[self.split_axis]]
        list_vals.append(self.split_vals)
        list_vals.append(self.end[self.split_axis])
        for i in range(num_pts):
            for j in range(len(list_vals) -1):
                if (pts[self.split_axis][i] >= list_vals[j]) and\
                   (pts[self.split_axis][i] < list_vals[j+1]):
                    indices[j].append(i)
        split_pts = []
        for j in range(len(list_vals) -1):
            split_pts.append(pts[:, indices[j]])
        return split_pts

    def __str__(self):
        str_val = "Cube ID: " + str(self.id_string) + "\n"
        str_val += "Boundary: "
        for i in range(self.dim):
            str_val += " [" + str(self.start[i]) + ", " + str(self.end[i]) + "]"
            if i < self.dim -1:
                str_val += " x"
            else:
                str_val += "\n"
        if self.split_axis != -1:
                str_val += "Axis: " + str(self.split_axis) + ", "
                str_val += "Split Values: " + str(self.split_vals)
        return str_val

    def print_cube(self):
        print_list = [self]
        while print_list:
            cube = print_list.pop(0)
            print(str(cube))
            print_list.extend(cube.child)
