import cv2
import numpy as np

def draw_bounding_box(image_path, detections):
    # Load the image
    image = cv2.imread(image_path)

    # Iterate through detections and draw bounding boxes
    for detection in detections:
        if detection['label'] == 'helmet':
            bbox = detection['bbox']
            # Convert normalized coordinates to pixel values
            height, width = image.shape[:2]
            x1 = int(bbox[0] * width)
            y1 = int(bbox[1] * height)
            x2 = int(bbox[2] * width)
            y2 = int(bbox[3] * height)

            # Draw the bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box

    # Display the image with bounding boxes
    cv2.imshow('Image with Bounding Box', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
detections = [{'label': 'helmet', 'bbox': [0.195, 0.233, 0.737, 0.537], 'score': 0.73}]
draw_bounding_box('2.jpeg', detections)