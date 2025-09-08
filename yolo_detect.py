import cv2
from ultralytics import YOLO
import numpy as np

# Load the trained YOLO model
model = YOLO('yolo_train/weights/best.pt')

# Loop to process frames
try:
    while True:
        # Wait for a set of frames from the camera

        # Use YOLOv5 for inference
        results = model('mujoco_tomato.png')

        # Process detection results and extract bounding boxes
        for result in results[0].boxes:
            # Get the coordinates (x, y, width, height) of the bounding box
            x_center, y_center, width, height = result.xywh[0]
            confidence = result.conf[0]
            class_id = result.cls[0]

            # Get class name from results
            class_name = results[0].names[int(class_id)]

            # Print the detected object's class and coordinates
            print(f"Object: {class_name}, Coordinates: ({x_center}, {y_center}, {width}, {height}), Confidence: {confidence:.2f}")

        # Draw bounding boxes and class labels on the frame
        frame = results[0].plot()

        # Display the result
        cv2.imshow('YOLOv5 Object Detection with Depth', frame)

        # Exit loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Stop the RealSense camera and close OpenCV window

    cv2.destroyAllWindows()
