import numpy as np
from math import ceil
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
import random
import plotly.graph_objects as go
from PIL import Image


# function to be removed
def line_draw(line, canv, size, color=(50,255,50)):
    
    def get_y(t):
        return -(line[0] * t + line[2]) / line[1]
    
    def get_x(t):
        return -(line[1] * t + line[2]) / line[0]
    
    w, h = size
    
    if line[0] != 0 and abs(get_x(0) - get_x(w)) < w:
        beg = (get_x(0), 0)
        end = (get_x(h), h)
    else:
        beg = (0, get_y(0))
        end = (w, get_y(w))
    
    canv.line([beg, end], width=4, fill=color)


def get_transformed_pixels_coords(I, H, shift=None):
    """Transforms pixel coordinates using a homography matrix.
    
    Args:
        I (numpy.ndarray): Input image.
        H (numpy.ndarray): Homography matrix.
        shift (tuple, optional): Shift values for x and y coordinates. Defaults to None.
    
    Returns:
        numpy.ndarray: Transformed pixel coordinates.
    """
    ys, xs = np.indices(I.shape[:2]).astype("float64")
    if shift is not None:
        ys += shift[1]
        xs += shift[0]
    ones = np.ones(I.shape[:2])
    coords = np.stack((xs, ys, ones), axis=2)
    coords_H = (H @ coords.reshape(-1, 3).T).T
    coords_H /= coords_H[:, 2, np.newaxis]
    cart_H = coords_H[:, :2]
    
    return cart_H.reshape((*I.shape[:2], 2))


def apply_H_fixed_image_size(I, H, corners):
    """Apply a given homography matrix to an image with fixed output size.
    
    Args:
        I (numpy.ndarray): Input image.
        H (numpy.ndarray): Homography matrix.
        corners (list): List of corner coordinates [xmin, xmax, ymin, ymax].
    
    Returns:
        numpy.ndarray: Transformed image with fixed size as uint8.
    """
    h, w = I.shape[:2]
    
    # corners
    c1 = np.array([1, 1, 1])
    c2 = np.array([w, 1, 1])
    c3 = np.array([1, h, 1])
    c4 = np.array([w, h, 1])
    
    # transformed corners
    Hc1 = H @ c1
    Hc2 = H @ c2
    Hc3 = H @ c3
    Hc4 = H @ c4
    Hc1 = Hc1 / Hc1[2]
    Hc2 = Hc2 / Hc2[2]
    Hc3 = Hc3 / Hc3[2]
    Hc4 = Hc4 / Hc4[2]
    
    xmin = corners[0]
    xmax = corners[1]
    ymin = corners[2]
    ymax = corners[3]

    size_x = ceil(xmax - xmin + 1)
    size_y = ceil(ymax - ymin + 1)
    
    # transform image
    H_inv = np.linalg.inv(H)
    
    out = np.zeros((size_y, size_x, 3))
    shift = (xmin, ymin)
    interpolation_coords = get_transformed_pixels_coords(out, H_inv, shift=shift)
    interpolation_coords[:, :, [0, 1]] = interpolation_coords[:, :, [1, 0]]
    interpolation_coords = np.swapaxes(np.swapaxes(interpolation_coords, 0, 2), 1, 2)
    
    out[:, :, 0] = map_coordinates(I[:, :, 0], interpolation_coords)
    out[:, :, 1] = map_coordinates(I[:, :, 1], interpolation_coords)
    out[:, :, 2] = map_coordinates(I[:, :, 2], interpolation_coords)
    
    return out.astype("uint8")


def Normalization(x):
    '''
    Normalize coordinates to centroid at origin and mean distance of sqrt(2).
    
    Parameters
    ----------
    x : numpy.ndarray
        Data to be normalized (3 x N array).

    Returns
    -------
    Tr : numpy.ndarray
        Transformation matrix (translation plus scaling).
    x : numpy.ndarray
        Transformed data.
    '''
    # Convert input to numpy array
    x = np.asarray(x)
    
    # Divide by the homogeneous coordinate to bring points to the Euclidean plane
    x = x / x[2, :]
    
    # Calculate mean and standard deviation of normalized coordinates
    m, s = np.mean(x, 1), np.std(x)
    
    # Scale factor to achieve mean distance of sqrt(2)
    s = np.sqrt(2) / s
    
    # Transformation matrix (translation plus scaling)
    Tr = np.array([[s, 0, -s * m[0]], [0, s, -s * m[1]], [0, 0, 1]])
    
    # Apply transformation to input data
    xt = Tr @ x
    
    return Tr, xt

def DLT_homography(points1, points2):
    """Compute the Direct Linear Transform (DLT) homography matrix between two sets of corresponding points.
    
        Args:
            points1 (numpy.ndarray): Array of shape (3, n) representing the first set of points.
            points2 (numpy.ndarray): Array of shape (3, n) representing the second set of points.
    
        Returns:
            numpy.ndarray: The 3x3 homography matrix that transforms points from the first image to the second image.
    """
    # Normalize the input points
    T1, points1n = Normalization(points1)
    T2, points2n = Normalization(points2)
    
    # Initialize the list to store the rows of the matrix A
    A = []
    n = points1.shape[1]

    # Fill the matrix A
    for i in range(n):
        x, y, z = points1n[0, i], points1n[1, i], points1n[2, i]
        u, v, w = points2n[0, i], points2n[1, i], points2n[2, i]
        A.append( [0, 0, 0, -w*x, -w*y, -w*z, v*x, v*y, v*z] )
        A.append( [w*x, w*y, w*z, 0, 0, 0, -u*x, -u*y, -u*z] )
        A.append( [-v*x, -v*y, -v*z, u*x, u*y, u*z, 0, 0, 0] )

    # Convert A to a numpy 2D array, i.e. a numpy matrix
    A = np.asarray(A)

    # Perform Singular Value Decomposition (SVD) on A
    U, d, Vt = np.linalg.svd(A)

    # Extract homography by retrieving the last row of Vt and normalizing it
    L = Vt[-1, :] / Vt[-1, -1]
    H = L.reshape(3, 3)
    
    # Denormalize H
    H = np.linalg.inv(T2) @ H @ T1
    
    return H
