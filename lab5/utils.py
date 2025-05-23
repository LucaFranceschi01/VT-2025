import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from PIL import Image
import random
import math
import sys
from tqdm.notebook import tqdm

def line_draw(line, canv, size):
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
    canv.line([beg, end], width=4)


def plot_img(img, do_not_use=[0]):
    plt.figure(do_not_use[0])
    do_not_use[0] += 1
    plt.imshow(img)


def optical_center(P):
    U, d, Vt = np.linalg.svd(P)
    o = Vt[-1, :3] / Vt[-1, -1]
    return o


def view_direction(P, x):
    # Vector pointing to the viewing direction of a pixel
    # We solve x = P v with v(3) = 0
    v = np.linalg.inv(P[:, :3]) @ np.array([x[0], x[1], 1])
    return v


def plot_camera(P, w, h, fig, legend, scale=1):

    o = optical_center(P)

    p1 = o + view_direction(P, [0, 0]) * scale
    p2 = o + view_direction(P, [w, 0]) * scale
    p3 = o + view_direction(P, [w, h]) * scale
    p4 = o + view_direction(P, [0, h]) * scale

    x = np.array([p1[0], p2[0], o[0], p3[0], p2[0], p3[0], p4[0], p1[0], o[0], p4[0], o[0], (p1[0] + p2[0]) / 2])
    y = np.array([p1[1], p2[1], o[1], p3[1], p2[1], p3[1], p4[1], p1[1], o[1], p4[1], o[1], (p1[1] + p2[1]) / 2])
    z = np.array([p1[2], p2[2], o[2], p3[2], p2[2], p3[2], p4[2], p1[2], o[2], p4[2], o[2], (p1[2] + p2[2]) / 2])

    fig.add_trace(go.Scatter3d(x=x, y=z, z=-y, mode="lines", name=legend))

    return


def plot_camera_col(P, w, h, fig, legend, col, scale=1):

    o = optical_center(P)

    p1 = o + view_direction(P, [0, 0]) * scale
    p2 = o + view_direction(P, [w, 0]) * scale
    p3 = o + view_direction(P, [w, h]) * scale
    p4 = o + view_direction(P, [0, h]) * scale

    x = np.array([p1[0], p2[0], o[0], p3[0], p2[0], p3[0], p4[0], p1[0], o[0], p4[0], o[0], (p1[0] + p2[0]) / 2])
    y = np.array([p1[1], p2[1], o[1], p3[1], p2[1], p3[1], p4[1], p1[1], o[1], p4[1], o[1], (p1[1] + p2[1]) / 2])
    z = np.array([p1[2], p2[2], o[2], p3[2], p2[2], p3[2], p4[2], p1[2], o[2], p4[2], o[2], (p1[2] + p2[2]) / 2])

    fig.add_trace(go.Scatter3d(x=x, y=z, z=-y, mode="lines", line=go.scatter3d.Line(color=f"rgb({col})"), name=legend))

    return

def correspondences_between_keypoints(kp_a, kp_b, matches_ab):
    pointsa_Hab = []
    pointsb_Hab = []
    for m in matches_ab:
        pointsa_Hab.append([kp_a[m.queryIdx].pt[0], kp_a[m.queryIdx].pt[1], 1])
        pointsb_Hab.append([kp_b[m.trainIdx].pt[0], kp_b[m.trainIdx].pt[1], 1])
        
    pointsa_Hab = np.asarray(pointsa_Hab)
    pointsa_Hab = pointsa_Hab.T
    pointsb_Hab = np.asarray(pointsb_Hab)
    pointsb_Hab = pointsb_Hab.T

    return pointsa_Hab, pointsb_Hab

def Normalization(x):

    x = np.asarray(x)
    x = x / x[2, :]

    m, s = np.mean(x, 1), np.std(x)
    s = np.sqrt(2) / s

    Tr = np.array([[s, 0, -s * m[0]], [0, s, -s * m[1]], [0, 0, 1]])

    xt = Tr @ x

    return Tr, xt


def compute_inliers(F, x1, x2, th):

    # Project the points into epipolar lines
    Fx1 = F @ x1
    Ftx2 = F.T @ x2

    n = x1.shape[1]
    x2tFx1 = np.zeros((1, n))

    # For each point, calculate epipolar constraint
    for i in range(n):
        x2tFx1[0, i] = x2[:, i].T @ F @ x1[:, i]

    # Evaluate distances (how close is a point to its epipolar line)
    den = Fx1[0, :] ** 2 + Fx1[1, :] ** 2 + Ftx2[0, :] ** 2 + Ftx2[1, :] ** 2
    den = den.reshape((1, n))

    d = x2tFx1**2 / den

    # If the distance to epipolar lines are relatively small (<th) then is inlier
    inliers_indices = np.where(d[0, :] < th)

    return inliers_indices[0]

def fundamental_matrix(points1, points2):

    # Normalize points in both images
    T1, points1n = Normalization(points1)
    T2, points2n = Normalization(points2)

    # TODO: compute the fundamental matrix F
    N = points1n.shape[1]

    # Compute A
    A = np.zeros((N, 9))
    for i in range(N):
        x = points1n[:, i]
        y = points2n[:, i]
        A[i, :] = np.array([y[0]*x[0], y[0]*x[1], y[0], y[1]*x[0], y[1]*x[1], y[1], x[0], x[1], 1])
    
    # Solve Af = 0
    _, _, Vh = np.linalg.svd(A) # check it comes transposed
    f = Vh[-1, :]

    # Reshape F from column array to matrix
    F = np.reshape(f, (3, 3))

    # Impose sigma_3 = 0 --> ideal case
    U, D, Vh = np.linalg.svd(F)
    D = np.diag(D)
    D[-1, -1] = 0
    F = U @ D @ Vh

    assert(np.linalg.matrix_rank(F) == 2)

    # Denormalize
    F = T2.T @ F @ T1

    return F

def ransac_fundamental_matrix(points1, points2, th, min_iterations):

    Ncoords, Npts = points1.shape

    p = 0.99
    # TODO: Select the number of correspondences to sample
    s = 8

    it = 0
    N = min_iterations
    best_inliers = np.empty(1)
    best_N = 1e16

    # Initialize tqdm with dynamic total
    pbar = tqdm(total=N, desc="RANSAC iterations", unit="iter")
    
    while it < N:

        # TODO: Randomly select s correspondences to compute F and its inliers
        indices = random.sample(range(Npts), s)
        F = fundamental_matrix(points1[:, indices], points2[:, indices])
        inliers = compute_inliers(F, points1, points2, th)

        # update estimate of iterations (the number of trials) to ensure we pick, with probability p,
        # an initial data set with no outliers
        w = inliers.shape[0] / Npts
        pOutlier = 1 - w**s
        eps = sys.float_info.epsilon
        pOutlier = max(eps, pOutlier)  # avoid log(0) that would cause a division by -Inf
        pOutlier = min(1 - eps, pOutlier)  # avoid log(1) that would cause a division by 0
        estimated_N = math.log(1 - p) / math.log(pOutlier)
        best_N = min(best_N, estimated_N)  # Keep the best (lowest) N
        N = max(min_iterations, best_N)  # Ensure at least min_iterations

        # TODO: Update the best inliers

        if len(inliers) >= len(best_inliers):
            print(f"it:{it} - best inliers: {len(best_inliers)} - w: {w:.2f} - N: {N:.5g}")
            best_inliers = inliers

        if int(N) > pbar.total:
            pbar.total = int(N)
            pbar.refresh()

        pbar.update(1)
        it += 1

    pbar.close()

    print(f"Number of iterations: {it}, best inliers: {len(best_inliers)} - N: {N:.5g}")

    # Recompute F from all the inliers
    F = fundamental_matrix(points1[:, best_inliers], points2[:, best_inliers])
    inliers_recomputed = compute_inliers(F, points1, points2, th)
    print(f"Recomputed inliers: {len(inliers_recomputed)}")

    return F, best_inliers


def triangulate(x1, x2, P1, P2, imsize):

    # only one point
    if x1.ndim == 1:
        x1 = np.array([x1]).T
        x2 = np.array([x2]).T

    # number of points
    n = x1.shape[1]

    # Normalization
    x1 = x1 / x1[2, :]
    x2 = x2 / x2[2, :]

    nx = imsize[0]
    ny = imsize[1]

    H = [[2 / nx, 0, -1], [0, 2 / ny, -1], [0, 0, 1]]

    x1_norm = H @ x1
    x2_norm = H @ x2
    P1_norm = H @ P1
    P2_norm = H @ P2

    # TODO: compute the triangulated points
    A = np.empty((4, 4))
    X = np.empty((4, n))
    for i in range(n): # no fancy way without for loop ?
        A[0, :] = np.array([x1_norm[0, i]*P1_norm[2, :] - P1_norm[0, :]])
        A[1, :] = np.array([x1_norm[1, i]*P1_norm[2, :] - P1_norm[1, :]])
        A[2, :] = np.array([x2_norm[0, i]*P2_norm[2, :] - P2_norm[0, :]])
        A[3, :] = np.array([x2_norm[1, i]*P2_norm[2, :] - P2_norm[1, :]])

        _, _, Vt = np.linalg.svd(A)

        # NOTE: no need to unnormalize
        X[:, i] = Vt[-1, :]

    return X


def camera_projection_matrix(F : np.ndarray,
                             K : np.ndarray,
                             nx : int,
                             ny : int,
                             points1 : np.ndarray,
                             points2 : np.ndarray,
                             indices_inlier_matches : np.ndarray):
    '''
    Computes all camera projection matrices from the fundamental matrix F and intrinsics K. Then
    it iterates over all possibilities, discarding impossible results.

    In this implementation the first camera is always in the center of the reference system.

    Parameters
    ----------
    F : (3x3) np.ndarray
        Fundamental matrix that relates the two views
    K : (3x3) np.ndarray
        Intrinsic parameters of the cameras
    nx: Width of the image
    ny: Height of the image
    points1: correspondences from first image to second
    points2: correspondences from second image to first
    indices_inlier_matches: inlier matches indices
    '''
    P1 = K @ np.eye(3, 4)

    E = K.T @ F @ K
    
    U, D, Vt = np.linalg.svd(E)

    # The SVD of E has several ambiguities. In particular, U and V may be
    # improper rotations in which case we need to change their sign.
    if np.linalg.det(U) < 0:
        U = -U

    if np.linalg.det(Vt) < 0:
        Vt = -Vt

    W = np.array([[0, -1, 0],
                [1, 0, 0],
                [0, 0, 1]])

    # TODO: Find the four potential camera projection matrices for the second camera
    Pc2 = np.empty(shape=(4, 3, 4))
    Pc2[0] = K @ np.concatenate((U @ W @ Vt, U[:, -1].reshape(3, 1)), axis=1)
    Pc2[1] = K @ np.concatenate((U @ W @ Vt, -U[:, -1].reshape(3, 1)), axis=1)
    Pc2[2] = K @ np.concatenate((U @ W.T @ Vt, U[:, -1].reshape(3, 1)), axis=1)
    Pc2[3] = K @ np.concatenate((U @ W.T @ Vt, -U[:, -1].reshape(3, 1)), axis=1)

    x1 = points1[:, indices_inlier_matches]
    x2 = points2[:, indices_inlier_matches]

    # We iterate over all 3 possible cameras (fixing the first one with R = I, t = 0)
    for P2i in Pc2:

        Xi = triangulate(x1[:, 0], x2[:, 0], P1, P2i, [nx, ny])
        Xi = Xi / Xi[3, :]

        x1est = P1 @ Xi
        x2est = P2i @ Xi

        # If the triangulated points are in front of both cameras
        if (x1est[2] > 0) and (x2est[2] > 0):
            P2 = P2i
            break

    return P2