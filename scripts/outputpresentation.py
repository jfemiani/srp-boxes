# given cx, cy, ux, uy, d, find 4 points
from math import hypot


class OutputRepresentations:
    def __init__(self, cx=0, cy=0, ux=0, uy=0, vd=0):
        self.cx = cx
        self.cy = cy
        self.ux = ux
        self.uy = uy
        self.vd = vd

    @staticmethod
    def from_rot_length_width(ctr, deg, length, width):
        cx, cy = ctr
        uax = cos(radians(deg))
        uay = sin(radians(deg))
        ux = uax * length / 2.
        uy = uay * length / 2.
        vd = width / 2.

        return OutputRepresentations(cx, cy, ux, uy, vd)

    @property
    def ud(self):
        return hypot(self.ux, self.uy)

    @property
    def u_vector(self):
        return np.array([self.ux, self.uy])

    @property
    def u_axis(self):
        return np.array([self.ux, self.uy]) / self.ud

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
        u = ((p[0] + p[3]) - (p[1] + p[2])) / 2.
        angle = arctan2(u[1], u[0])
        angle = (angle + 2 * pi) % (pi / 2.)
        uax = cos(angle)
        uay = sin(angle)
        assert uax >= 0 and uay >= 0

        ud = p.dot([uax, uay]).max()
        vd = p.dot([-uay, uax]).max()
        ux, uy = ud * np.array([uax, uay])

        return OutputRepresentations(ctr[0], ctr[1], ux, uy, vd)