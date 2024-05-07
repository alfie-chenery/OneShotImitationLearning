import numpy
import cv2
import os
from matplotlib import pyplot as plt


class FeatureMatcher():

    def __init__(self):
        self.sift = None
        self.orb = None
        self.orbgms = None  #Although GMS still uses ORB features, the parameers need to be very different to get good results 

        self.siftKnnMatcher = None
        self.siftFlannMatcher = None
        self.orbBfMatcher = None



    def OpenImage(self, img_path):
        return cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)


    def SIFTFeatures(self, img):
        """
        Param img: cv2 image object
        Return keypoints: List[keypoint object] containing x,y,size,orientation
               descriptors: numpy array of size (numKeypoints, 128)
        """
        #TODO consider post processing keypoints to make them a nicer form to work with

        if self.sift is None:
            self.sift = cv2.SIFT_create()

        return self.sift.detectAndCompute(img, None)
    

    def ORBFeatures(self, img):
        """
        Param img: cv2 image object
        Return keypoints: 
               descriptors: 
        """
        #TODO doc

        if self.orb is None:
            self.orb = cv2.ORB_create()
            
        return self.orb.detectAndCompute(img, None)
    

    def ORBGMSFeatures(self, img):
        """
        Param img: cv2 image object
        Return keypoints: 
               descriptors: 
        """
        #TODO doc

        if self.orbgms is None:
            self.orbgms = cv2.ORB_create(10000, fastThreshold=0)
            
        return self.orbgms.detectAndCompute(img, None)
    

    def MatchKeypoints_SIFT_KNN(self, img1, img2, numMatches=None):
        """
        TODO doc
        """

        if self.siftKnnMatcher is None:
            self.siftKnnMatcher = cv2.BFMatcher()

        _, des1 = self.SIFTFeatures(img1)
        _, des2 = self.SIFTFeatures(img2)

        matches = self.siftKnnMatcher.knnMatch(des1,des2, k=2)

        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append(m)

        # Sort by distance so closest match is first
        good = sorted(good, key = lambda x:x.distance)

        if numMatches is not None:
            good = good[:numMatches]
        
        return good
    

    def MatchKeypoints_SIFT_FLANN(self, img1, img2, numMatches=None):
        """
        TODO doc
        """

        if self.siftFlannMatcher is None:
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks=50) # or pass empty dictionary
            self.siftFlannMatcher = cv2.FlannBasedMatcher(index_params,search_params)

        _, des1 = self.SIFTFeatures(img1)
        _, des2 = self.SIFTFeatures(img2)

        
        matches = self.siftFlannMatcher.knnMatch(des1,des2,k=2)

        # Apply ratio test
        good = []
        for m,n in matches:
            if m.distance < 0.75*n.distance:
                good.append(m)

        # Sort by distance so closest match is first
        good = sorted(good, key = lambda x:x.distance)

        if numMatches is not None:
            good = good[:numMatches]
        
        return good


    def MatchKeypoints_ORB_BF(self, img1, img2, numMatches=None):
        """
        TODO doc
        """

        if self.orbBfMatcher is None:
            self.orbBfMatcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        _, des1 = self.ORBFeatures(img1)
        _, des2 = self.ORBFeatures(img2)


        #TODO: fix this
        if len(des1) == 0 or len(des2) == 0:
            return []

        matches = self.orbBfMatcher.match(des1, des2)
 
        # Sort by distance so closest match is first
        matches = sorted(matches, key = lambda x:x.distance)

        if numMatches is not None:
            matches = matches[:numMatches]

        return matches
    

    def MatchKeypoints_ORB_GMS(self, img1, img2, numMatches=None):
        """
        TODO doc
        """

        if self.orbBfMatcher is None:
            self.orbBfMatcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        kp1, des1 = self.ORBGMSFeatures(img1)
        kp2, des2 = self.ORBGMSFeatures(img2)

        # #TODO: fix this
        # if len(des1) == 0 or len(des2) == 0:
        #     return []

        matches = self.orbBfMatcher.match(des1, des2)
        matches_gms = cv2.xfeatures2d.matchGMS(img1.shape, img2.shape, kp1, kp2, matches, withRotation=True, withScale=True)
 
        # Sort by distance so closest match is first
        matches_gms = sorted(matches_gms, key = lambda x:x.distance)

        if numMatches is not None:
            matches_gms = matches_gms[:numMatches]

        return matches_gms


    def findMatches(self, img1, img2):
        """
        Wrapper function for use in main.py. Allows for easily changing this function
        to change which features are used and test for best perfromance
        """
        pass




def main(img1_path, img2_path, method, numMatches=None):
    fm = FeatureMatcher()
    dir_path = dir_path = os.path.dirname(os.path.realpath(__file__))

    img1 = fm.OpenImage(img1_path)
    img2 = fm.OpenImage(img2_path)

    if(method == "SIFT_KNN"):
        kp1, _ = fm.SIFTFeatures(img1)
        kp2, _ = fm.SIFTFeatures(img2)
        matches = fm.MatchKeypoints_SIFT_KNN(img1, img2, numMatches=numMatches)

    elif(method == "SIFT_FLANN"):
        kp1, _ = fm.SIFTFeatures(img1)
        kp2, _ = fm.SIFTFeatures(img2)
        matches = fm.MatchKeypoints_SIFT_FLANN(img1, img2, numMatches=numMatches)
    
    elif(method == "ORB_BF"):
        kp1, _ = fm.ORBFeatures(img1)
        kp2, _ = fm.ORBFeatures(img2)
        matches = fm.MatchKeypoints_ORB_BF(img1, img2, numMatches=numMatches)

    elif(method == "ORB_GMS"):
        kp1, _ = fm.ORBGMSFeatures(img1)
        kp2, _ = fm.ORBGMSFeatures(img2)
        matches = fm.MatchKeypoints_ORB_GMS(img1, img2, numMatches=numMatches)
        print(matches)


    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3)
    plt.show()


if __name__ == "__main__":
    dir_path = dir_path = os.path.dirname(os.path.realpath(__file__))
    img1_path = dir_path + "\\temp\\live-rgb.jpg"
    img2_path = dir_path + "\\temp\\initial_scene-rgb.jpg"

    main(img1_path, img2_path, "ORB_GMS", numMatches=None)
    
    

