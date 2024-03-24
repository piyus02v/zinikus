import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import torch

class ObjectDetection(Node):
    def __init__(self):
        super().__init__('object_detection')
        self.bridge = CvBridge()
        self.model = YOLO('yolov8s-world.pt')
        self.model.set_classes(['bottle'])
        self.model.to("cpu")
        #self.video_path = 'img/vid1.mp4'
        self.vid = cv2.VideoCapture(0)
        self.img_sub = self.create_subscription(Image,'/camera1/image_raw',self.image_callback,10)

    def image_callback(self,data):
        frame = self.bridge.imgmsg_to_cv2(data)
        if frame is not None:
            results = self.model.predict(frame)
            annotated_frame = results[0].plot()
            cv2.imshow('image',annotated_frame)
            cv2.waitKey(0)
            boxes = results[0].boxes.xyxy.tolist()

            # Perform further processing with the detected objects
            self.move_objects_to_container(boxes)
    def move_objects_to_container(self, boxes):
        pass

def main(args=None):
    rclpy.init(args=args)
    object_detection = ObjectDetection()
    rclpy.spin(object_detection)
    object_detection.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
