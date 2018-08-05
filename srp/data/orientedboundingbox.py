"""
A rotated rectangle.

"""
from __future__ import division
import numpy as np
import shapely
import shapely.geometry
from matplotlib.pylab import plt


class OrientedBoundingBox(object):
    """
    An Oriented Bounding Box

    This is a rectangle with a rotation.

    * Due to symmetry, the 'rotation' is always between 0 and 90 degrees.
    
    * The box has two axes; a u_axis and a v_axis. The u_axis is a unit vector
      whose angle in the 2D plane is between 0 and 90 degrees.
    
    * The v_axis is a 90 degree counter-clockwise rotation of the u_axis.

    Examples
    --------

    A horizontal 3 x 1 box

    >>> b1 = OrientedBoundingBox.from_points([[0, 0], [3, 0], [3, 1], [0, 1]])
    >>> print(b1.u_axis)
    [1. 0.]
    >>> print(b1.v_axis + 0)  # add xero to avoid -0 as a result.
    [0. 1.]
    >>> b1.u_length
    1.5
    >>> b1.v_length
    0.5
    >>> print(b1.origin)
    [1.5 0.5]


    A rotated box
    
    >>> b2 = OrientedBoundingBox.from_rot_length_width((1, 2), 30, 3, 1)
    >>> b2.u_length
    1.5
    >>> b2.v_length
    0.5
    >>> print(b2.origin)
    [1. 2.]
    >>> print(b2.u_axis)
    [0.866... 0.5 ]

    You can compare shapes using intersection over union
    
    >>> b1.iou(b2) > 0.99
    False
    >>> b1.iou(b1) > 0.99
    True

    You can use the ``.shape`` method to convert this to a shapely polygon
    
    >>> isinstance(b2.shape(), shapely.geometry.Polygon)
    True

    # You can conveniently include these in matplotlib plots as well.
    
    .. code-block:: python

        rasterio.plot(some_basemap_image)
        b2.plot(alpha=0.5)
    
    Attributes
    ----------

    _center (ndarray): The center of the box.
    _u_vector (ndarray): A vector perpendicular to one of the axes of the box. The length of ``_u_vector`` is the distance of that edge from the center of the box. 
    _v_length (float): The distance from the center to the edges of the box that a parallel to ``_u_vector``. 


    """

    def __init__(self, cx=0, cy=0, ux=0, uy=0, v_length=0):
        # pylint:disable=too-many-arguments
        """__init__

        :param cx: The center
        :param cy:
        :param ux: The u_vector
        :param uy:
        :param v_length: The length of the v_vector(direction is inferred from the u_vector)
        """
        self._center = np.array([cx, cy], dtype=float)
        self._u_vector = np.array([ux, uy], dtype=float)
        self._center.flags.writeable = False
        self._u_vector.flags.writeable = False
        self._v_length = v_length

    @staticmethod
    def from_rot_length_width(ctr, deg, length, width):
        """Construct a OBB from an angle (degrees), and the length in the u and v directions.

        Note that the OBB may exchange u and v directions to keep the angle in 0--90

        :param ctr: Center of the OrientedBoundingBox
        :param deg: Angle of the u_vector(degrees). If this is not in 0-90 then an
                    equivalent OBB is returned with an angle that is within that range.
        :param length: The length of the u_vector
        :param width: The length of the v_vector
        """
        cx, cy = ctr
        uax = np.cos(np.radians(deg))
        uay = np.sin(np.radians(deg))
        ux = uax * length / 2.
        uy = uay * length / 2.
        v_length = width / 2.

        return OrientedBoundingBox(cx, cy, ux, uy, v_length)


#     def rot_length_width(self):
#         angle = np.degrees(np.arctan2(self.uy, self.ux))
#         return np.array([self.cx, self.cy, angle, 2*self.u_length, 2*self.v_length])

    @property
    def angle(self):
        """The angle between the X axis an the box/s u_vector. 
        
        This is always between 0 and 90 degrees.
        """
        return np.degrees(np.arctan2(self.u_vector[1], self.u_vector[0]))

    @property
    def u_length(self):
        """The radius (half the width) of the box in the u-direction.
        
        Returns:
            float: The length of the u_vector. 
        """
        return np.linalg.norm(self.u_vector)

    @property
    def v_length(self):
        """v_length"""
        return self._v_length

    @property
    def u_vector(self) -> np.ndarray:
        """u_vector"""
        return self._u_vector

    @property
    def u_axis(self) -> np.ndarray:
        """The u_axis of the box (a unit vector).

        Returns:
           ndarray: A 2D unit vector with x > y 
        """
        return self._u_vector / self.u_length

    @property
    def v_axis(self) -> np.ndarray:
        """v_axis (unit vector)"""
        uax, uay = self.u_axis
        return np.array([-uay, uax])

    @property
    def v_vector(self) -> np.ndarray:
        """v_vector"""
        return self.v_axis * self.v_length

    @property
    def origin(self) -> np.ndarray:
        """origin (aka center...)"""
        return self._center

    def vectors(self) -> np.ndarray:
        """vectors from the center to each corner in CCW order."""
        # pylint:disable=invalid-unary-operand-type
        return np.array([
            self.u_vector + self.v_vector, self.v_vector - self.u_vector, -self.u_vector - self.v_vector,
            self.u_vector - self.v_vector
        ])

    def points(self):
        """points in CCW order"""
        return self.vectors() + self.origin

    @staticmethod
    def rot_length_width_from_points(points):
        """
        return cx, cy, deg, length(2*u_length), width(2*v_length)
        """

        obb = OrientedBoundingBox.from_points(points)
        return np.array([obb.origin[0], obb.origin[1], obb.angle, 2 * obb.u_length, 2 * obb.v_length])


    @staticmethod
    def from_points(points):
        """Construct an OrientedBoundingBox from points.

        :param points: The points at four corners.

        NOTE: We assume that the points are from a rectangle (now skew or tapering).
              If there is some doubt, consider passing:
              ```
              shapely.geometry.Polygon(points).minimum_rotated_rectangle.coords
              ```
        """
        points = np.asanyarray(points)
        ctr = points.mean(0)
        vectors = points - ctr
        u = ((vectors[0] + vectors[3]) - (vectors[1] + vectors[2])) / 2.
        angle = np.arctan2(u[1], u[0])
        angle = (angle + 2 * np.pi) % (np.pi / 2.)
        uax = np.cos(angle)
        uay = np.sin(angle)
        assert uax >= 0 and uay >= 0

        u_length = vectors.dot([uax, uay]).max()
        v_length = vectors.dot([-uay, uax]).max()
        ux, uy = u_length * np.array([uax, uay])

        return OrientedBoundingBox(ctr[0], ctr[1], ux, uy, v_length)

    def shape(self):
        """Covert to a shapely Polygon object.
        """
        return shapely.geometry.Polygon(self.points())

    def iou(self, other):
        """Compute the intersection over union (iou) with another OBB

        :param other: another OBB.
        """
        shape = self.shape()
        other = other.shape()
        result = shape.intersection(other).area / other.union(shape).area
        return result

    def plot(self, ax=None, **kwargs):
        """plot using matplotlib

        :param ax: The axis to plot on (default is plt.gca())
        :param **kwargs: Additional keyword arguments passed on to plt.Polygon.
        """
        if ax is None:
            ax = plt.gca()

        patch_args = dict(alpha=0.5, fill=False, color='r')
        patch_args.update(kwargs)
        ax.add_patch(plt.Polygon(self.points(), **patch_args))
