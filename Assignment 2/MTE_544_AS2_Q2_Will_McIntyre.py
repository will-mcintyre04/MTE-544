"""
Student Name & Last Name: Will McIntyre
Original Author : Pi Thanacha Choopojcharoen
You must change the name of your file to MTE_544_AS2_Q2_(your full name).py
Do not use jupyter notebook.

*You may want to install the following libraries if you haven't done so.*

pip install numpy matplotlib pandas scipy

"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
import random
import pandas as pd

def plot_ellipse(S, b, ax):
    ## updated the code here to get the eigenvalues
    ## and then calcualte the semi-axis lengths
    ## using 1 / sqrt(eigval)
    eigvals, eigvecs = np.linalg.eigh(S)
    aa, bb = 1 / np.sqrt(eigvals)

    theta = np.linspace(0, 2 * np.pi, 100)
    unit_circle = np.array([np.cos(theta), np.sin(theta)])
    ellipse_param = np.array([aa * unit_circle[0], bb * unit_circle[1]])
    ellipse_points = eigvecs @ ellipse_param + b[:, np.newaxis]

    ax.plot(ellipse_points[0, :], ellipse_points[1, :], 'b-', label='Fitted Ellipse')
    ax.plot(b[0], b[1], 'ro', label='Center')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend(loc='best')
    ax.grid(True)

def visualize_data(p, ax, inliers, threshold):
    ax.scatter(p[:, 0], p[:, 1], color='red', alpha=0.5, label='Raw Measurements (Ellipse)')
    ax.scatter(inliers[:, 0], inliers[:, 1], color='purple', alpha=0.7, label='Inliers')

    for point in inliers:
        circle = plt.Circle(point, threshold, color='orange', fill=False, linestyle='--', alpha=0.7)
        ax.add_patch(circle)

def is_positive_definite(matrix):
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False

def fit_ellipse_subset(points):
    # Given 5 points, construct the ellipse

    x = points[:, 0]
    y = points[:, 1]

    # construct matrix M and constact vector k
    M = np.column_stack((x**2, 2*x*y, y**2, -2*x, -2*y))
    k = -np.ones(5)

    try:
        # solve for the coefficients:
        v = np.linalg.solve(M, k)
    except np.linalg.LinAlgError:
        return None, None

    # extract coefficients
    A, B, C, D, E = v

    M_ABC = np.array([[A, B], [B, C]])
    M_DE = np.array([D, E])

    try:
        # Compute alpha, Q, and b using the Part 1 formulas
        alpha = 1 / ((M_DE.T @ np.linalg.solve(M_ABC, M_DE)) - 1)
        Q = alpha * M_ABC
        b = np.linalg.solve(Q, alpha * M_DE)
        return Q, b
    except np.linalg.LinAlgError:
        return None, None
    

def ransac_ellipse(data, num_iterations=1000, threshold=0.8):
    best_Q = None
    best_b = None
    best_inliers = np.array([])
    max_inliers_count = 0

    for _ in range(num_iterations):
        ## get random indices
        indices = np.random.choice(len(data), size=5, replace=False)
        subset = data[indices]

        # fit to ellipse
        Q, b = fit_ellipse_subset(subset)

        # Ensure Q is valid and represents an ellipse (positive definite)
        if Q is None or not is_positive_definite(Q):
            continue

        # reject centers that are far away from the data
        # data is all around -20 and 20
        if np.linalg.norm(b) > 20:
            continue

        # vector form of (p_i - b) for all points
        p_minus_b = data - b

        # calculate distance using equation from assignment
        dist_sq = np.sum((p_minus_b @ Q) * p_minus_b, axis=1)

        # prevent negative
        dist_sq = np.maximum(dist_sq, 0)
        dist = np.sqrt(dist_sq)

        # apply the inlier inequality condition
        inlier_mask = np.abs(dist - 1) <= threshold
        current_inliers = data[inlier_mask]

        # update best model if the current one has more inliers
        if len(current_inliers) > max_inliers_count:
            max_inliers_count = len(current_inliers)
            best_Q = Q
            best_b = b
            best_inliers = current_inliers

    return best_Q, best_b, best_inliers

if __name__ == "__main__":
    # Load the data from CSV file and select N random points
    N = 1000
    all_data = pd.read_csv('data_x_y.csv').to_numpy()
    dataset = all_data[np.random.choice(all_data.shape[0], N, replace=False), :]
    # dataset is p
    
    Q, b_est, inliers = ransac_ellipse(dataset)
    
    # Plot the raw measurements and fitted ellipse
    fig, ax1 = plt.subplots(1, 1, figsize=(16, 8))
    visualize_data(dataset, ax1, inliers, threshold=0.08)
    plot_ellipse(Q, b_est, ax1)
    ax1.set_title("RANSAC Ellipse Fitting with Threshold Visualization")

    plt.show()