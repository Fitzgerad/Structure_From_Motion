from fundamental_matrix import fundamental_matrix
from global_config import *
import numpy as np
from points_utils import *
import cv2

class essential_matrix():
    def __init__(self):
        self.F = fundamental_matrix()

    def getEMatrix(self, Fundamental_matrix):
        EMatrix = INTRINSIC_MATRIX.T.dot(Fundamental_matrix.dot(INTRINSIC_MATRIX))
        U, sigma, V = np.linalg.svd(EMatrix)
        print(sigma)
        sigma[2] = 0
        sigma[0] = (sigma[0] + sigma[1]) / 2
        sigma[1] = sigma[0]
        EMatrix = U.dot(np.diag(sigma)).dot(V)
        EMatrix = cv2.normalize(EMatrix, EMatrix, 1, 0)
        return EMatrix

    def projectImg(self, base_index, proj_index):
        Fundamental_matrix, inliers, kp1, kp2 = self.F.projectImg(base_index, proj_index)
        Essential_matrix = self.getEMatrix(Fundamental_matrix)
        # print(Essential_matrix)
        # initial_points = np.array([kp1[match.queryIdx].pt for match in inliers])
        # projected_points = np.array([kp2[match.trainIdx].pt for match in inliers])
        # print(cv2.findEssentialMat(initial_points, projected_points, INTRINSIC_MATRIX))
        if type(Essential_matrix) == int:
            return None
        self.esitimatePosition(Essential_matrix, inliers, kp1, kp2, base_index)

    def esitimatePosition(self, Essential_matrix, inliers, kp1, kp2, base_index):
        U, sigma, V = np.linalg.svd(Essential_matrix)
        C1 = U[:, 2]
        C2 = np.negative(C1)
        R1 = U.dot(W_MATRIX).dot(V)
        # print(R1, np.linalg.det(R1))
        if np.linalg.det(R1) < 0:
            R1 = np.negative(R1)
        # print(R1, np.linalg.det(R1))
        R2 = U.dot(W_MATRIX.T).dot(V)
        if np.linalg.det(R2) < 0:
            R2 = np.negative(R2)
        alternatives = [[C1, R1], [C1, R2],
                        [C2, R1], [C2, R2]]
        errors = [0, 0, 0, 0]
        for i in range(4):
            errors[i] = self.computeErrors(alternatives[i][0], alternatives[i][1], inliers, kp1, kp2)
        print(len(inliers), errors)
        C, R = alternatives[errors.index(min(errors))]
        print(C, R)
        points = self.getPoints(C, R, inliers, kp1, kp2)
        self.showPoints(points, inliers, kp1, base_index)
        # P1 = INTRINSIC_MATRIX.dot(np.c_[np.identity(3), np.array([0, 0, 0])])
        # P2 = INTRINSIC_MATRIX.dot(R.dot(np.c_[np.identity(3), np.negative(C)]))
        # initial_points = np.array([kp1[match.queryIdx].pt for match in inliers])
        # projected_points = np.array([kp2[match.trainIdx].pt for match in inliers])
        # X = cv2.triangulatePoints(P1[:3], P2[:3], initial_points.T, projected_points.T)
        # pcwrite("leile.ply", (X/X[3])[0:3].T)
        return points

    def showPoints(self, points, inliers, kp1, base_index):
        points = np.array(points)
        rgb = []
        for match in inliers:
            x, y = int(kp1[match.queryIdx].pt[0]), int(kp1[match.queryIdx].pt[1])
            rgb.append(self.F.images[base_index][y][x])
        rgb = np.array(rgb)
        pcwrite_rgb("test.ply", points, rgb)

    def computeErrors(self, C, R, inliers, kp1, kp2):
        P1 = INTRINSIC_MATRIX.dot(np.c_[np.identity(3), np.array([0, 0, 0])])
        P2 = INTRINSIC_MATRIX.dot(R.dot(np.c_[np.identity(3), np.negative(C)]))
        errors = 0
        for match in inliers:
            x, y = kp1[match.queryIdx].pt[0], kp1[match.queryIdx].pt[1]
            u, v = kp2[match.trainIdx].pt[0], kp2[match.trainIdx].pt[1]
            A = np.r_[x*P1[2] - P1[0], y*P1[2] - P1[1],
                      u*P2[2] - P2[0], v*P2[2] - P2[1]]
            U, sigma, V = np.linalg.svd(A)
            X = V[:, -1]
            for i in range(4):
                X[i] /= X[3]
            if R[2].dot((X[0:3][0] - C)) < 0:
                errors += 1
        return errors

    def getPoints(self, C, R, inliers, kp1, kp2):
        P1 = INTRINSIC_MATRIX.dot(np.c_[np.identity(3), np.array([0, 0, 0])])
        P2 = INTRINSIC_MATRIX.dot(R.dot(np.c_[np.identity(3), np.negative(C)]))
        points = []
        for match in inliers:
            x, y = kp1[match.queryIdx].pt[0], kp1[match.queryIdx].pt[1]
            u, v = kp2[match.trainIdx].pt[0], kp2[match.trainIdx].pt[1]
            A = np.r_[x*P1[2] - P1[0], y*P1[2] - P1[1],
                      u*P2[2] - P2[0], v*P2[2] - P2[1]]
            U, sigma, V = np.linalg.svd(A)
            X = V[:, -1]
            for i in range(4):
                X[i] /= X[3]
            points.append(X[0:3])
        return points

E = essential_matrix()
print(E.projectImg(1, 0))