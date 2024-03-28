#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped
import sys

class motion_model:
    def __init__(self):
        self.x = 0
        self.y = 0

        self.old_x = 0
        self.old_y = 0

        self.delta_x = 0
        self.delta_y = 0

        self.VAR = 0.05

        self.ball_pose_sub = rospy.Subscriber("ball_pose", PoseWithCovarianceStamped, self.ball_callback)
        self.ball_prediction_pub = rospy.Publisher("ball_prediction", PoseWithCovarianceStamped, queue_size=10)

    def ball_callback(self, data):
        self.old_x = self.x
        self.old_y = self.y

        self.x = data.pose.pose.position.x
        self.y = data.pose.pose.position.y

        self.delta_x = self.x - self.old_x
        self.delta_y = self.y - self.old_y

        self.predicted_x = self.x + self.delta_x
        self.predicted_y = self.y + self.delta_y

        self.message = PoseWithCovarianceStamped()
        self.message.header.frame_id = "d400_link"
        self.message.header.stamp = rospy.Time.now()
        self.message.pose.pose.position.x = self.predicted_x
        self.message.pose.pose.position.y = self.predicted_y
        self.message.pose.covariance[0] = self.VAR
        self.message.pose.covariance[3] = self.VAR

        self.ball_prediction_pub.publish(self.message)

def main(args):
    rospy.init_node("motion_model", anonymous=True)
    detector = motion_model()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main(sys.argv)