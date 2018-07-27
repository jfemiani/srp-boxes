from __future__ import division
# given cx, cy, ux, uy, d, find 4 points
from builtins import object
from past.utils import old_div
import numpy as np
from math import hypot
import shapely
import matplotlib as plt

class OrientedBoundingBox(object):
    def __init__(self, cx=0, cy=0, ux=0, uy=0, vd=0):
        self.cx = cx
        self.cy = cy
        self.ux = ux
        self.uy = uy
        self.vd = vd

    @staticmethod
    def from_rot_length_width(ctr, deg, length, width):
        cx, cy = ctr
        uax = np.cos(np.radians(deg))
        uay = np.sin(np.radians(deg))
        ux = uax * length / 2.
        uy = uay * length / 2.
        vd = old_div(width, 2.)

        return OrientedBoundingBox(cx, cy, ux, uy, vd)
    
    @staticmethod
    def get_rot_length_width_from_points(points):
        """
        return cx, cy, deg, length(2*ud), width(2*vd)
        """
        
        oo = OrientedBoundingBox.from_points(points)
        angle =  np.degrees(np.arctan2(oo.uy, oo.ux))
        return np.array([oo.cx, oo.cy, angle, 2*oo.ud, 2*oo.vd])
    
#     def rot_length_width(self):
#         angle = np.degrees(np.arctan2(self.uy, self.ux))
#         return np.array([self.cx, self.cy, angle, 2*self.ud, 2*self.vd])
    
    @property
    def ud(self):
        return hypot(self.ux, self.uy)

    @property
    def u_vector(self):
        return np.array([self.ux, self.uy])

    @property
    def u_axis(self):
        return old_div(np.array([self.ux, self.uy]), self.ud)

    @property
    def v_axis(self):
        uax, uay = self.u_axis
        return np.array([-uay, uax])

    @property
    def v_vector(self):
        return self.v_axis * self.vd

    @property
    def origin(self):
        return np.array([self.cx, self.cy])

    def vectors(self):
        return np.array([
            self.u_vector + self.v_vector,
            -self.u_vector + self.v_vector,
            -self.u_vector - self.v_vector,
            self.u_vector - self.v_vector
        ])

    def points(self):
        return self.vectors() + self.origin

    @staticmethod
    def from_points(points):
        ctr = points.mean(0)
        p = points - ctr
        u = old_div(((p[0] + p[3]) - (p[1] + p[2])), 2.)
        angle = np.arctan2(u[1], u[0])
        angle = (angle + 2 * np.pi) % (old_div(np.pi, 2.))
        uax = np.cos(angle)
        uay = np.sin(angle)
        assert uax >= 0 and uay >= 0

        ud = p.dot([uax, uay]).max()
        vd = p.dot([-uay, uax]).max()
        ux, uy = ud * np.array([uax, uay])

        return OrientedBoundingBox(ctr[0], ctr[1], ux, uy, vd)
    
    def shape(self):
        return shapely.geometry.Polygon(self.points())
    
    def plot(self, ax):
        if ax is None:
            ax = plt.gca()
            
        ax.add_patch(plt.Polygon(self.points(), alpha=0.5, fill=False,  color='r'))
        