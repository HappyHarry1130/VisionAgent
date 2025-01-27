from vision_agent.tools import load_image, owlv2_object_detection
import cv2 

def check_helmets(image_path):
    image = load_image(image_path)
    # Detect people and helmets, filter out the lowest confidence helmet score of 0.15
    detections = owlv2_object_detection("person, chair", image)
    height, width = image.shape[:2]

    # Separate people and helmets
    people = [d for d in detections if d['label'] == 'person']
    tables = [d for d in detections if d['label'] == 'chair']
    print(people)
    print(tables)

    people_with_helmets = 0
    people_without_helmets = 0

    for person in people:
        person_x = (person['bbox'][0] + person['bbox'][2]) / 2
        person_y = person['bbox'][1]  # Top of the bounding box

        helmet_found = False
        for helmet in tables:
            helmet_x = (helmet['bbox'][0] + helmet['bbox'][2]) / 2
            helmet_y = (helmet['bbox'][1] + helmet['bbox'][3]) / 2

            # Check if the helmet is within 20 pixels of the person's head. Unnormalize
            # the coordinates so we can better compare them.
            if (abs((helmet_x - person_x) * width) < 20 and
                -5 < ((helmet_y - person_y) * height) < 20):
                helmet_found = True
                break

        if helmet_found:
            people_with_helmets += 1
        else:
            people_without_helmets += 1

    # Draw bounding boxes for helmets on the image
    for helmet in tables:
        bbox = helmet['bbox']
        x1 = int(bbox[0] * width)
        y1 = int(bbox[1] * height)
        x2 = int(bbox[2] * width)
        y2 = int(bbox[3] * height)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box for helmets

    # Draw bounding boxes for persons on the image
    for person in people:
        bbox = person['bbox']
        x1 = int(bbox[0] * width)
        y1 = int(bbox[1] * height)
        x2 = int(bbox[2] * width)
        y2 = int(bbox[3] * height)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue box for persons

    # Display the image with bounding boxes
    cv2.imshow('Image with Bounding Boxes', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return {  # Change this line
        "people_with_helmets": people_with_helmets,
        "people_without_helmets": people_without_helmets
    }

print(check_helmets('5423_group-of-people-sitting-in-a-cafe.jpg'))