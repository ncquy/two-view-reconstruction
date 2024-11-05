import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load camera parameters
fx, fy = 1086, 1086
cx, cy = 512, 384
k1 = -0.0568965  # Distortion coefficient

# Camera matrix K
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0,  1]])

# Load images
img1 = cv2.imread('hw5_data/003.jpg')
img2 = cv2.imread('hw5_data/005.jpg')

# Load 2D points from CSV files
points1 = np.loadtxt('hw5_data/003.csv', delimiter=',')
points2 = np.loadtxt('hw5_data/005.csv', delimiter=',')

# Step 1: Visualize 2D points on images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
plt.scatter(points1[:, 0], points1[:, 1], color='red', s=10)
plt.title("Image 1 with 2D points")

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
plt.scatter(points2[:, 0], points2[:, 1], color='red', s=10)
plt.title("Image 2 with 2D points")
plt.show()

# Step 2: Calculate the Essential matrix
E, _ = cv2.findEssentialMat(points1, points2, K)
print("Essential Matrix:\n", E)

# Step 3: Recover pose (R, t) from the essential matrix
_, R, t, _ = cv2.recoverPose(E, points1, points2, K)
print("Rotation Matrix (R):\n", R)
print("Translation Vector (t):\n", t)

# Step 4: Triangulate points to reconstruct 3D structure
# Set projection matrices for two views
P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
P2 = np.hstack((R, t))

# Use K to get the full projection matrices
P1 = K @ P1
P2 = K @ P2

# Triangulate points
points4D_hom = cv2.triangulatePoints(P1, P2, points1.T, points2.T)
points3D = points4D_hom[:3] / points4D_hom[3]

# Step 5: Visualize 3D points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points3D[0], points3D[1], points3D[2], c='b', marker='o')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title("3D Reconstruction from Two Views")
plt.show()
