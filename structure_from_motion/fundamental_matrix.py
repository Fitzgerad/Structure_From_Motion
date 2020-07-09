import cv2
import os
import random
import math
import numpy as np
from global_config import *

class fundamental_matrix():
    def __init__(self):
        self.match = cv2.BFMatcher()
        self.sift = cv2.xfeatures2d.SIFT_create()
        self.images = []
        if (not self.loadImages()):
            raise ('There\'s no valid image in the DIR_PATH!')
        self.size = len(self.images)

    def loadImages(self):
        if os.path.exists(DIR_PATH):
            image_names = os.listdir(DIR_PATH)
            for image_name in image_names:
                image_path = os.path.join(DIR_PATH, image_name)
                self.images.append(cv2.imread(image_path))
            return True
        else:
            return False

    def getFeatures(self, img):
        temp_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        temp_kp, temp_des = self.sift.detectAndCompute(temp_img, None)
        return temp_kp, temp_des

    def matchFeatures(self, base_des, proj_des):
        matches = self.match.knnMatch(base_des, proj_des, k=2)
        index = 0
        while index < len(matches):
            if matches[index][0].distance * 1.2 >= matches[index][1].distance:
                matches.pop(index)
            else:
                matches[index] = matches[index][0]
                index += 1
        return matches

    def getFMatrix(self, initial_points, projected_points):
        A = []
        b = []
        # Figure out the fundamental matrix omitting rank=2
        for i in range(len(initial_points)):
            x, y, z = initial_points[i][0], initial_points[i][1], initial_points[i][2]
            u, v, w = projected_points[i][0], projected_points[i][1], projected_points[i][2]
            A.append([x*u, x*v, x, y*u, y*v, y, u, v])
            b.append(-1)
        FMatrix = list(np.linalg.lstsq(A, b, rcond=-1)[0])
        FMatrix.append(1)
        FMatrix = np.reshape(FMatrix, (3, 3)).T
        # subject the matrix to the rank=2 condition
        U, sigma, V = np.linalg.svd(FMatrix)
        sigma[2] = 0
        FMatrix = U.dot(np.diag(sigma)).dot(V)
        return FMatrix

    def ranSAC(self, matches, kp1, kp2):
        # 判断共线点数
        F_best = None
        I_most = 0
        I_best = []
        iter_times = int(math.log(0.01) / math.log(1 - pow(0.8, FUNDAMENTAL_RANSAC_SIZE)))
        for k in range(iter_times):
            indexs = random.sample(range(len(matches)), FUNDAMENTAL_RANSAC_SIZE)
            initial_points = [kp1[matches[indexs[i]].queryIdx].pt + (1,) for i in range(FUNDAMENTAL_RANSAC_SIZE)]
            projected_points = [kp2[matches[indexs[i]].trainIdx].pt + (1,) for i in range(FUNDAMENTAL_RANSAC_SIZE)]
            Fundamental_matrix = self.getFMatrix(initial_points, projected_points)
            inliers = self.getInliers(kp1, kp2, matches, Fundamental_matrix)
            if len(inliers) <= FUNDAMENTAL_RANSAC_SIZE:
                continue
            initial_points = [kp1[match.queryIdx].pt + (1,) for match in inliers]
            projected_points = [kp2[match.trainIdx].pt + (1,) for match in inliers]
            Fundamental_matrix = self.getFMatrix(initial_points, projected_points)
            inliers = self.getInliers(kp1, kp2, matches, Fundamental_matrix)
            if len(inliers) > I_most:
                F_best = Fundamental_matrix
                I_most = len(inliers)
                I_best = inliers
        # points1 = np.array([kp1[matches[i].queryIdx].pt + (1,) for i in range(len(matches))])
        # points2 = np.array([kp2[matches[i].trainIdx].pt + (1,) for i in range(len(matches))])
        # F_best = cv2.findFundamentalMat(points2,points1)[0]
        # I_best = self.getInliners(kp1, kp2, matches, F_best)
        # print(F_best)
        return F_best, I_best

    def getInliers(self, kp1, kp2, matches, Fundamental_matrix):
        inliers = []
        for match in matches:
            error = np.dot(kp2[match.trainIdx].pt + (1,),
                           np.dot(Fundamental_matrix, np.array(kp1[match.queryIdx].pt + (1,)).T))
            # print(error)
            if abs(error) < FUNDAMENTAL_RANSAC_THRESHOLD:
                inliers.append(match)
        return inliers

    def projectImg(self, base_index, proj_index):
        kp1, des1 = self.sift.detectAndCompute(self.images[base_index], None)
        kp2, des2 = self.sift.detectAndCompute(self.images[proj_index], None)
        matches = self.matchFeatures(des1, des2)
        if len(matches) >= 300:
            # indexs = random.sample(range(len(matches)), FUNDAMENTAL_RANSAC_SIZE)
            # initial_points = [kp1[matches[indexs[i]].queryIdx].pt + (1,) for i in range(FUNDAMENTAL_RANSAC_SIZE)]
            # projected_points = [kp2[matches[indexs[i]].trainIdx].pt + (1,) for i in range(FUNDAMENTAL_RANSAC_SIZE)]
            # self.matrix[base_index][proj_index] = self.getFMatrix(initial_points, projected_points)
            # inliners = self.getInliners(kp1, kp2, matches, self.matrix[base_index][proj_index])
            # self.showInliners(self.images[base_index], self.images[proj_index], kp1, kp2, inliners)
            Fundamental_matrix, inliers = self.ranSAC(matches, kp1, kp2)
            self.showInliners(self.images[base_index], self.images[proj_index], kp1, kp2, inliers)
        return Fundamental_matrix, inliers, kp1, kp2

    def showInliners(self, img1, img2, kp1, kp2, inliners):
        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                          singlePointColor = None,
                          flags = 2)
        cv2.namedWindow('Match Image', cv2.WINDOW_NORMAL)
        img3 = cv2.drawMatches(img1, kp1, img2, kp2, inliners, None, **draw_params)
        cv2.imshow("Match Image", img3)
        cv2.resizeWindow("Match Image", 1800, 500)
        cv2.waitKey(0)
        return

# FF = fundamental_matrix()
# print(FF.projectImg(8, 9))
