import cv2
import numpy as np
from fundamental_matrix import fundamental_matrix
from global_config import *
from points_utils import *

class essential_matrix():
    def __init__(self):
        self.F = fundamental_matrix()

    def getEMatrix(self, Fundamental_matrix):
        EMatrix = INTRINSIC_MATRIX.T.dot(Fundamental_matrix.dot(INTRINSIC_MATRIX))
        U, sigma, V = np.linalg.svd(EMatrix)
        # print(sigma)
        sigma[2] = 0
        sigma[0] = (sigma[0] + sigma[1]) / 2
        sigma[1] = sigma[0]
        EMatrix = U.dot(np.diag(sigma)).dot(V)
        EMatrix = cv2.normalize(EMatrix, EMatrix, 1, 0)
        # x = EMatrix
        # x -= np.mean(x, axis=0)
        # x /= np.std(x, axis=0)
        # return x
        return EMatrix

    def projectImg(self, base_index, proj_index):
        Fundamental_matrix, inliers, kp1, kp2 = self.F.projectImg(base_index, proj_index)
        Essential_matrix = self.getEMatrix(Fundamental_matrix)
        # initial_points = np.array([kp1[match.queryIdx].pt for match in inliers])
        # projected_points = np.array([kp2[match.trainIdx].pt for match in inliers])
        # Essential_matrix = cv2.findEssentialMat(initial_points, projected_points, INTRINSIC_MATRIX)[0]
        if type(Essential_matrix) == int:
            return None
        return self.esitimatePosition(Essential_matrix, inliers, kp1, kp2, base_index)

    def esitimatePosition(self, Essential_matrix, inliers, kp1, kp2, base_index):
        '''
        估计相机位置参数
        '''
        # initial_points = np.array([kp1[match.queryIdx].pt for match in inliers])
        # projected_points = np.array([kp2[match.trainIdx].pt for match in inliers])
        # points, R, t, mask = cv2.recoverPose(Essential_matrix, initial_points, projected_points, INTRINSIC_MATRIX)
        # P1 = INTRINSIC_MATRIX.dot(np.c_[np.identity(3), np.array([0, 0, 0])])
        # P2 = INTRINSIC_MATRIX.dot(R.dot(np.c_[np.identity(3), np.negative(t)]))
        # X = cv2.triangulatePoints(P1[:3], P2[:3], initial_points.T, projected_points.T)
        # pcwrite("leile.ply", (X / X[3])[0:3].T)

        U, sigma, V = np.linalg.svd(Essential_matrix)
        U = np.array(U)
        V = np.array(V)
        C1 = U[:, 2]
        C2 = np.negative(C1)
        R1 = U.dot(W_MATRIX).dot(V)
        if np.linalg.det(R1) < 0:
            R1 = np.negative(R1)
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

        # print(C, R)
        points = self.getPoints(C, R, inliers, kp1, kp2)
        self.showPoints(points, inliers, kp1, base_index)
        return points,C,R

    def showPoints0(self, points):
        points = np.array(points)
        pcwrite("statue.ply", points)

    def showPoints(self, points, inliers, kp1, base_index):
        '''
        将三维点写入文件
        '''
        points = np.array(points)
        # print(points)
        rgb = []
        for match in inliers:
            x, y = int(kp1[match.queryIdx].pt[0]), int(kp1[match.queryIdx].pt[1])
            rgb.append(self.F.images[base_index][y][x])
        rgb = np.array(rgb)
        pcwrite_rgb(str(base_index)+".ply", points, rgb)

    def computeErrors(self, C, R, inliers, kp1, kp2):
        '''
        计算当前 C,R 的不符合条件的点数
        '''
        errors = 0
        points = self.getPoints(C, R, inliers, kp1, kp2)
        for X in points:
            if R[2].dot((X[0:3] - C)) < 0 or X[2] < 0:
                errors += 1
        return errors

    def getPoints(self, C, R, inliers, kp1, kp2):
        '''
        获取点的三维坐标
        '''
        P1 = INTRINSIC_MATRIX.dot(np.c_[np.identity(3), np.array([0, 0, 0])])
        P2 = INTRINSIC_MATRIX.dot(R.dot(np.c_[np.identity(3), np.negative(C)]))
        points = []
        for match in inliers:
            x, y = kp1[match.queryIdx].pt[0], kp1[match.queryIdx].pt[1]
            u, v = kp2[match.trainIdx].pt[0], kp2[match.trainIdx].pt[1]
            A = np.vstack(((y * P1[2] - P1[1]),
                      (-x * P1[2] + P1[0]),
                      (v * P2[2] - P2[1]),
                      (-u * P2[2] + P2[0])))
            A1 = A[:, 0:3]
            b = A[:, 3]
            X = np.linalg.lstsq(A1, b, rcond=-1)[0].tolist()
            X[1] *= -1
            points.append(X[0:3])
        return points

def trans(points, t, R):
    '''
    转换三维点的坐标系
    '''
    res = list()
    for p in points:
        res.append((R.dot(p)+t).T)
    return np.array(res)

def combine():
    '''
    将相邻图片对得到的结果拼接在一起
    '''
    E = essential_matrix()
    num = E.F.size
    pp, t, R = E.projectImg(0, 1)
    pp = trans(pp, t, R)
    for i in range(1,num-1):
        p, t, R = E.projectImg(i,i+1)
        p = np.vstack((p,pp))
        pp = trans(p, t, R)
        print(i)
    E.showPoints0(pp)

def main():
    E = essential_matrix()
    num = E.F.size
    combine()
    # for i in range(0, num-1):
    #     E.projectImg(i,i+1)
    #     print (i)

if __name__ == '__main__':
    main()

