import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load camera parameters
fx, fy = 1086, 1086
cx, cy = 512, 384
k1 = -0.056896

# Camera matrix K
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0,  1]])

# Images
img1 = cv2.imread('hw5_data/003.jpg')
img2 = cv2.imread('hw5_data/005.jpg')

# 2D points
points1 = np.loadtxt('hw5_data/003.csv', delimiter=',')
points2 = np.loadtxt('hw5_data/005.csv', delimiter=',')

# a. 2D point visualization on two images 
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

# b. The essential matrix between two images 
E, _ = cv2.findEssentialMat(points1, points2, K)
print("Essential Matrix:\n", E)

# c. The rotation and translation, R and t
_, R, t, _ = cv2.recoverPose(E, points1, points2, K)
print("Rotation Matrix (R):\n", R)
print("Translation Vector (t):\n", t)

# d. 3D visualization of reconstructed 3D points in more than two viewpoints
# Triangulate points to reconstruct 3D structure
# Set projection matrices for two views
P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
P2 = np.hstack((R, t))

# Use K to get the full projection matrices
P1 = K @ P1
P2 = K @ P2

# Triangulate points
points4D_hom = cv2.triangulatePoints(P1, P2, points1.T, points2.T)
points3D = points4D_hom[:3] / points4D_hom[3]
points3D = points3D

# 3D points viulization
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points3D[0], points3D[1], points3D[2], c='b', marker='o')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title("3D Reconstruction from Two Views")
plt.show()
