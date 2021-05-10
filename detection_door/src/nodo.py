#! /usr/bin/env python

import rospy
import sys
from geometry_msgs.msg import Twist
from math import pi
import time

def move():
    # Starts a new node
    rospy.init_node('nodo', anonymous=True)
    velocity_publisher = rospy.Publisher('/pepper/cmd_vel', Twist, queue_size=10)
    vel_msg = Twist()
    while not rospy.is_shutdown():
        print('inizio')
        vel_msg.linear.x = 0
        vel_msg.linear.y = 0
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = 1.57
        velocity_publisher.publish(vel_msg)
        #time.sleep(1)
        #vel_msg.linear.x = 0
        #vel_msg.linear.y = 0
        #vel_msg.linear.z = 0
        #vel_msg.angular.x = 0
        #vel_msg.angular.y = 0
        #vel_msg.angular.z = 0
        #velocity_publisher.publish(vel_msg)
        print('fine')
        






if __name__ == '__main__':
    move()
