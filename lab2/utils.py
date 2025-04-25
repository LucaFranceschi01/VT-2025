import math
import random
import sys
from operator import itemgetter

import numpy as np
import plotly.graph_objects as go


def Normalization(x):

    x = np.asarray(x)
    x = x / x[2, :]

    m, s = np.mean(x, 1), np.std(x)
    s = np.sqrt(2) / s

    Tr = np.array([[s, 0, -s * m[0]], [0, s, -s * m[1]], [0, 0, 1]])

    xt = Tr @ x

    return Tr, xt


def DLT_homography(points1, points2):

    # Normalize points in both images
    T1, points1n = Normalization(points1)
    T2, points2n = Normalization(points2)

    A = []
    n = points1.shape[1]

    for i in range(n):
        x, y, z = points1n[0, i], points1n[1, i], points1n[2, i]
        u, v, w = points2n[0, i], points2n[1, i], points2n[2, i]
        A.append([0, 0, 0, -w * x, -w * y, -w * z, v * x, v * y, v * z])
        A.append([w * x, w * y, w * z, 0, 0, 0, -u * x, -u * y, -u * z])
        A.append([-v * x, -v * y, -v * z, u * x, u * y, u * z, 0, 0, 0])

    # Convert A to array
    A = np.asarray(A)

    U, d, Vt = np.linalg.svd(A)

    # Extract homography (last line of Vt)
    L = Vt[-1, :] / Vt[-1, -1]
    H = L.reshape(3, 3)

    # Denormalise
    H = np.linalg.inv(T2) @ H @ T1

    return H


def Normalise_last_coord(x):
    xn = x / x[2, :]

    return xn


def Inliers(H, points1, points2, th):

    # Check that H is invertible
    if abs(math.log(np.linalg.cond(H))) > 15:
        idx = np.empty(1)
        return idx

    # transformed points (in both directions)
    H_points1 = H @ points1
    Hinv_points2 = np.linalg.inv(H) @ points2

    # normalise homogeneous coordinates (third coordinate to 1)
    points1 = Normalise_last_coord(points1)
    points2 = Normalise_last_coord(points2)
    H_points1 = Normalise_last_coord(H_points1)
    Hinv_points2 = Normalise_last_coord(Hinv_points2)

    # compute the symmetric geometric error
    d2 = np.sum((points1 - Hinv_points2) ** 2 + (points2 - H_points1) ** 2, axis=0)

    inliers_indices = np.where(d2 < th**2)

    return inliers_indices[0]


def Ransac_DLT_homography(points1, points2, th, max_it):

    Ncoords, Npts = points1.shape

    it = 0
    best_inliers = np.empty(1)

    while it < max_it:
        indices = random.sample(range(1, Npts), 4)
        H = DLT_homography(points1[:, indices], points2[:, indices])
        inliers = Inliers(H, points1, points2, th)

        # test if it is the best model so far
        if inliers.shape[0] > best_inliers.shape[0]:
            best_inliers = inliers

        # update estimate of iterations (the number of trials) to ensure we pick, with probability p,
        # an initial data set with no outliers
        fracinliers = inliers.shape[0] / Npts
        pNoOutliers = 1 - fracinliers**4
        eps = sys.float_info.epsilon
        pNoOutliers = max(eps, pNoOutliers)  # avoid division by -Inf
        pNoOutliers = min(1 - eps, pNoOutliers)  # avoid division by 0
        p = 0.99
        max_it = math.log(1 - p) / math.log(pNoOutliers)

        it += 1

    # compute H from all the inliers
    H = DLT_homography(points1[:, best_inliers], points2[:, best_inliers])
    inliers = best_inliers

    return H, inliers


def optical_center(P):
    U, d, Vt = np.linalg.svd(P)
    o = Vt[-1, :3] / Vt[-1, -1]
    return o


def optical_center_2(P):
    KR, p4 = P[:, :3], P[:, -1]
    KR_inv = np.linalg.inv(KR)
    o = -KR_inv @ p4
    return o


def view_direction(P, x):
    # Vector pointing to the viewing direction of a pixel
    # We solve x = P v with v(3) = 0
    v = np.linalg.inv(P[:, :3]) @ np.array([x[0], x[1], 1])
    return v


def plot_camera(P, w, h, fig, legend, scale=200):

    o = optical_center(P)
    # o = optical_center_2(P)

    # scale = 200
    p1 = o + view_direction(P, [0, 0]) * scale
    p2 = o + view_direction(P, [w, 0]) * scale
    p3 = o + view_direction(P, [w, h]) * scale
    p4 = o + view_direction(P, [0, h]) * scale

    edges = np.column_stack(
        (
            *(p1, p2, p3, p4, p1),  # rectangle
            *(o, p1, o, p2, o, p3, o, p4),  # optical centre to rectangle vertices
            *(o, (p1 + p2) / 2),  # optical centre to centre of top edge of rectangle
        )
    )

    x = edges[0, :]
    y = edges[1, :]
    z = edges[2, :]

    fig.add_trace(go.Scatter3d(x=x, y=z, z=-y, mode="lines", name=legend))

    return


def plot_image_origin(w, h, fig, legend):
    p1 = np.array([0, 0, 0])
    p2 = np.array([w, 0, 0])
    p3 = np.array([w, h, 0])
    p4 = np.array([0, h, 0])

    # p1 = np.array([-w / 2, -h / 2, 0])
    # p2 = np.array([w / 2, -h / 2, 0])
    # p3 = np.array([w / 2, h / 2, 0])
    # p4 = np.array([-w / 2, h / 2, 0])

    edges = np.column_stack([p1, p2, p3, p4, p1])

    x = edges[0, :]
    y = edges[1, :]
    z = edges[2, :]

    fig.add_trace(go.Scatter3d(x=x, y=z, z=-y, mode="lines", name=legend))

    return
