import numpy as np
from matplotlib import pyplot as plt
import math

CONSTR = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
plotting_type_of_strain = 'x'

def matrix_strain_to_DIC(matrix):
    """reshaping to [-1 x 2]"""
    return matrix.reshape(2, -1).T

def matrix_DIC_to_strain(matrix,shape):
    """reshaping from [-1 x 2] to rectangle"""
    return matrix.T.reshape(shape)

class Element():
    def __init__(self, shape, constr, el_num):
        self.p1n = np.array([math.floor(el_num / (shape[2] -1)), el_num % (shape[2]-1)])
        self.CONSTR=constr
        self.strain_x = 0
        self.strain_y = 0
        self.strain_a = 0
    def get_4pp(self):
        """get all points positions in grid matrix"""
        return np.concatenate([self.p1n]*4,axis=0).reshape(4,2) + self.CONSTR
    def plot_point(self, grid):
        plt.plot(grid[0, self.p1n[0], self.p1n[1]], grid[1, self.p1n[0], self.p1n[1]], 'co')
    def plot_points(self, grid):
        for coords in self.get_4pp():
            plt.plot(grid[0, coords[0], coords[1]], grid[1, coords[0], coords[1]], 'ro')
    def get_val_on_points(self, grid):
        return grid[:, self.get_4pp()[:, 0], self.get_4pp()[:, 1]].T
    def plot_element(self, grid):
        """plot element rectangle"""
        points = self.get_val_on_points(grid)
        for i in range (0,self.CONSTR.shape[0]):
            fir = i
            sec = (i+1)%self.CONSTR.shape[0]
            plt.plot([points[fir,0],points[sec,0]],[points[fir,1],points[sec,1]],'k-',alpha=0.2)
    def get_angle(self, grid, P1, PM, P2):
        """get angle between points"""
        points = self.get_val_on_points(grid)
        l1 = ((points[P1,0]-points[PM,0])**2+(points[P1,1]-points[PM,1])**2)**0.5
        l2 = ((points[P2,0]-points[PM,0])**2+(points[P2,1]-points[PM,1])**2)**0.5
        lp = ((points[P2,0]-points[P1,0])**2+(points[P2,1]-points[P1,1])**2)**0.5
        cosal = (l1**2+l2**2-lp**2)/2/l1/l2
        return math.acos(cosal)
    def strain(self, grid_bef, grid_act):
        """calc strain from deformations"""
        points_bef = self.get_val_on_points(grid_bef)
        points_act = self.get_val_on_points(grid_act)
        # lengths before
        lenx0 = (points_bef[2, 0] - points_bef[1, 0] + points_bef[3, 0] - points_bef[0, 0]) / 2
        leny0 = (points_bef[1, 1] - points_bef[0, 1] + points_bef[2, 1] - points_bef[3, 1]) / 2
        alpha0 = (np.pi / 2 - (self.get_angle(grid_bef, 3, 0, 1) + self.get_angle(grid_bef, 1, 2, 3))) / 2
        # lengths after
        lenx1 = (points_act[2, 0] - points_act[1, 0] + points_act[3, 0] - points_act[0, 0]) / 2
        leny1 = (points_act[1, 1] - points_act[0, 1] + points_act[2, 1] - points_act[3, 1]) / 2
        alpha1 = (np.pi / 2 - (self.get_angle(grid_act, 3, 0, 1) + self.get_angle(grid_act, 1, 2, 3))) / 2
        # strain
        self.strain_x = lenx1/lenx0-1
        self.strain_y = leny1/leny0-1
        self.strain_a = alpha1-alpha0
        self.strain_eqv = max(
            abs((self.strain_x+self.strain_y)/2 + math.sqrt(((self.strain_x-self.strain_y)**2/2**2+(2*self.strain_a/2)**2))),
            abs((self.strain_x+self.strain_y)/2 - math.sqrt(((self.strain_x-self.strain_y)**2/2**2+(2*self.strain_a/2)**2))))
        # return self.strain_x,self.strain_y,self.strain_a
    def set_strain_val(self):
        """actualising type strain to plotting"""
        self.strain_val = 0
        if plotting_type_of_strain == 'x':
            self.strain_val = self.strain_x
        elif plotting_type_of_strain == 'y':
            self.strain_val = self.strain_y
        elif plotting_type_of_strain == 'a':
            self.strain_val = self.strain_a
        elif plotting_type_of_strain == 'eqv':
            self.strain_val = self.strain_eqv
    def plot_strain(self, grid, min=-0.1, max=0.1):
        """plot element with color based on strain value"""
        # val=self.strain_eqv
        points = self.get_val_on_points(grid)[:, :].T.tolist()
        R = (self.strain_val-min)/(max-min)
        G = 0
        B = 1-R
        plt.fill(points[0],points[1],color=(R,G,B,1))

