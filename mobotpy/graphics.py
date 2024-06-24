"""
Python module graphics.py for drawing basic shapes.
Author: Joshua A. Marshall <joshua.marshall@queensu.ca>
GitHub: https://github.com/botprof/agv-examples
"""

import numpy as np

"""SHAPE FUNCTIONS"""


def draw_circle(x, y, radius, increment=10):
    """Finds points that approximate a circle as a regular polygon.

    The regular polygon has centre (x, y), a radius, and the function computes
    a point every angular increment [deg] (default 10).
    """
    angles = np.deg2rad(np.arange(0.0, 360.0, increment))
    X = radius * np.cos(angles)
    Y = radius * np.sin(angles)
    X += x
    Y += y
    return X, Y


def draw_rectangle(x, y, length, width, angle):
    """Finds points that draw a rectangle.

    The rectangle has centre (x, y), a length, width, and angle [rad].
    """
    V = np.zeros((2, 5))
    l = 0.5 * length
    w = 0.5 * width
    V = np.array([[-l, -l, l, l, -l], [-w, w, w, -w, -w]])
    R = np.array([[np.cos(angle), np.sin(-angle)],
                 [np.sin(angle), np.cos(angle)]])
    V = R @ V
    X = V[0, :] + x
    Y = V[1, :] + y
    return X, Y


def draw_triangle(x, y, length, height, angle):
    """Finds points that draw a triangle.

    The triangle has a vertex at (x, y), a length (long dimension), 
    height, and angle [rad].

    Negative height will invert the triangle. 
    Angles are positive counter-clockwise.

    The triangle is drawn as follows:

              /\
            /    \
          /        \          [height]
        /            \
      /                \
    /____________________\
          [length]       ^ [x,y], rotation point

    """
    V = np.zeros((2, 4))
    l = length
    h = height
    V = np.array([[0, -l, -l/2, 0], [0, 0, h, 0]])
    R = np.array([[np.cos(angle), np.sin(-angle)],
                 [np.sin(angle), np.cos(angle)]])
    V = R @ V
    X = V[0, :] + x
    Y = V[1, :] + y
    return X, Y
