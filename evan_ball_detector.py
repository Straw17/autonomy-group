#!/usr/bin/env python3

from cmath import sqrt
import math
from struct import calcsize
from turtle import pos, up
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import rospy
from sensor_msgs.msg import Image
import sys
import imutils
from visualization_msgs.msg import Marker
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import tf
from actionlib_msgs.msg import GoalID
from geometry_msgs.msg import Twist
from std_msgs.msg import Int16MultiArray
from geometry_msgs.msg import PoseWithCovarianceStamped

class ball_detector:
    def __init__(self):
        self.PURPLE_H_HIGH = 290
        self.PURPLE_S_HIGH = 40
        self.PURPLE_V_HIGH = 100
        self.PURPLE_H_LOW = 235
        self.PURPLE_S_LOW = 10
        self.PURPLE_V_LOW = 40

        self.GREEN_H_HIGH = 145
        self.GREEN_S_HIGH = 84
        self.GREEN_V_HIGH = 70
        self.GREEN_H_LOW = 110
        self.GREEN_S_LOW = 40
        self.GREEN_V_LOW = 25

        self.BALL_PURPLE = False

        self.MAX_RANGE_PURPLE = 2
        self.MAX_RANGE_GREEN = 6

        self.setParamsFromColor() #set the appropriate parameters based on what our color is

        self.BALL_RADIUS = 0.15 / 2

        self.FILTER_MAX = 13
        self.FILTER_MIN = 7

        self.position2D = [0,0]
        self.distance = 0
        self.angleY = 0
        self.range = 0

        self.MAX_ANGLE_Y = math.radians(69 / 2)
        self.CENTER_X = 640 / 2
        self.CENTER_Y = 480 / 2

        self.MAX_ANGLE_Y_VIEW = math.radians(69/2 - 3)

        self.RANGE_VAR_COEF = 0.005
        self.ANGLE_VAR_COEF = 45

        self.ballNotFound = True

        self.bridge = CvBridge()
        self.imageSub = rospy.Subscriber("/d400/color/image_raw", Image, self.color_callback)
        self.depthSub = rospy.Subscriber("/d400/aligned_depth_to_color/image_raw", Image, self.depth_callback)
        self.ball_pose_pub = rospy.Publisher("ball_pose", PoseWithCovarianceStamped, queue_size=10)

    def setParamsFromColor(self):
        if self.BALL_PURPLE:
            self.H_HIGH = self.PURPLE_H_HIGH
            self.S_HIGH = self.PURPLE_S_HIGH
            self.V_HIGH = self.PURPLE_V_HIGH
            self.H_LOW = self.PURPLE_H_LOW
            self.S_LOW = self.PURPLE_S_LOW
            self.V_LOW = self.PURPLE_V_LOW

            self.MAX_RANGE = self.MAX_RANGE_PURPLE
        else:
            self.H_HIGH = self.GREEN_H_HIGH
            self.S_HIGH = self.GREEN_S_HIGH
            self.V_HIGH = self.GREEN_V_HIGH
            self.H_LOW = self.GREEN_H_LOW
            self.S_LOW = self.GREEN_S_LOW
            self.V_LOW = self.GREEN_V_LOW

            self.MAX_RANGE = self.MAX_RANGE_GREEN

    def color_callback(self, data):
        try:
            cvImage = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError:
            pass

        cvImage = self.transformImage(cvImage)

        cv2.imshow("Color", cvImage)
        cv2.waitKey(3)

    def depth_callback(self, data):
        try:
            self.depthImage = self.bridge.imgmsg_to_cv2(data, "16UC1")
        except CvBridgeError:
            pass

    def transformImage(self, img):
        filterSize = self.FILTER_MAX

        while filterSize >= self.FILTER_MIN:
            copy = img.copy()

            #select colors in range
            transform = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h = 0.5 #adjust from 360 degrees to 180 degrees
            sv = 255.0/100.0 #adjust from % to 0-255 scale
            lowerBound = np.array([self.H_LOW * h, self.S_LOW * sv, self.V_LOW * sv])
            upperBound = np.array([self.H_HIGH * h, self.S_HIGH * sv, self.V_HIGH * sv])
            transform = cv2.inRange(transform, lowerBound, upperBound)

            #remove noise
            kernel = np.ones((filterSize,filterSize), np.uint8)
            transform = cv2.morphologyEx(transform, cv2.MORPH_CLOSE, kernel)
            transform = cv2.morphologyEx(transform, cv2.MORPH_OPEN, kernel)
            
            ##cv2.imshow("Transform", transform)
            ##cv2.waitKey(3)

            #find contours
            contours = cv2.findContours(transform, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(contours)

            transform = cv2.cvtColor(transform, cv2.COLOR_GRAY2RGB)

            #find the largest circle in the image
            biggestBall = [(0,0),0] #[pixel coords, radius]
            ballDetected = False
            for contour in contours:
                (x,y), r = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(r)

                ##cv2.circle(img, center, radius, (0, 0, 255), 5)
                ##cv2.circle(img, center, 0, (0, 0, 255), 5)

                if radius > biggestBall[1]:
                    ballDetected = True
                    biggestBall[0] = center
                    biggestBall[1] = radius

            if not ballDetected:
                filterSize -= 2 #try again with a less restrictive filter
            else:
                try:
                    self.distance = self.depthImage[biggestBall[0][1]][biggestBall[0][0]] / 1000.0 #get the corresponding pixel from the depth camera and convert to meters
                except IndexError:
                    pass

                self.position2D = biggestBall[0]
                self.calcPosition()

                cv2.circle(img, biggestBall[0], biggestBall[1], (0, 0, 255), 5)
                cv2.circle(img, biggestBall[0], 0, (0, 0, 255), 5)

                self.ballNotFound = False
                self.publishHandler()
                return img
            
        self.ballNotFound = True
        self.publishHandler()
        return img

    def publishHandler(self):
        mean = self.calcCartesianMean() #calculate pose

        #construct the message and set up the mean values
        message = PoseWithCovarianceStamped()
        message.header.frame_id = "d400_link"
        message.header.stamp = rospy.Time.now()
        message.pose.pose.position.x = mean[0]
        message.pose.pose.position.y = -mean[1]

        #if ball not detected, set the variance very high, using the calculated covariance matrix if it is detected
        if self.ballNotFound or self.outOfRange():
            message.pose.covariance[0] = 1000000 #set variance arbitrarily high
            message.pose.covariance[7] = 1000000

            print("No ball found")
        else:
            covMat = self.calcCartesianCov() #calculate covariance matrix

            message.pose.covariance[0] = covMat[0][0]
            message.pose.covariance[1] = -covMat[0][1]
            message.pose.covariance[6] = covMat[1][0]
            message.pose.covariance[7] = covMat[1][1]

            print(round(180/math.pi * self.angleY,3), round(self.range,3)) #print angle and range

        self.ball_pose_pub.publish(message) #publish message

    def outOfRange(self): #check if the ball is out of range
        if abs(self.angleY) <= self.MAX_ANGLE_Y_VIEW and self.range < self.MAX_RANGE and self.range > self.BALL_RADIUS:
            return False
        else:
            return True
        
    def calcCartesianMean(self):
        mean = [self.range * math.cos(self.angleY),
                self.range * math.sin(self.angleY)]
        
        return mean
        
    def calcCartesianCov(self):
        #set up the rotation matrix
        rot = ([math.cos(self.angleY), -math.sin(self.angleY)],
               [math.sin(self.angleY), math.cos(self.angleY)])
        
        #range variance scales quadratically, angle variance scales inversely with range, since it is based on the pixels the ball takes up
        cov = ([self.RANGE_VAR_COEF * self.range**2, 0],
               [0, (self.ANGLE_VAR_COEF / self.range) * (self.MAX_ANGLE_Y / self.CENTER_X)])
        #angle variance here is messed up, definitely needs to be changed

        #perform the rotation on the covariance matrix
        final = np.matmul(np.matmul(rot, cov), np.transpose(rot))

        return final

    def calcPosition(self): #calculate angleY and range
        self.angleY = self.MAX_ANGLE_Y * (self.position2D[0] - self.CENTER_X) / self.CENTER_X
        self.range = self.distance / math.cos(self.angleY) + self.BALL_RADIUS

def main(args):
    rospy.init_node("ball_detector", anonymous=True)
    detector = ball_detector()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(sys.argv)
