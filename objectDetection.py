# import the necessary packages
import cv2
import uuid
import os
import numpy as np

# import detectron2 packages
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.model_zoo import model_zoo

# import custom packages
from utils.jpgConverter import convertToJPG
from utils.compareImages import compare_images
from firebase_files.index import upload_to_firebase


# Load config and model
cfg = get_cfg()
cfg.MODEL.DEVICE = 'cpu'  # Set device to CPU
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = '/Users/pruthvipatel/Desktop/objectdetection/model_final_f10217.pkl'
predictor = DefaultPredictor(cfg)

# Set up webcam
# cv2.namedWindow("DroneCam", cv2.WINDOW_NORMAL)
cv2.startWindowThread()
cap = cv2.VideoCapture(1)

prev_images = []

while True:
    ret, frame = cap.read()

    # Run object detection on the frame
    outputs = predictor(frame)

    # Filter out all other classes except for "person"
    instances = outputs["instances"]
    keep = (instances.pred_classes == 0) | (instances.pred_classes == 16)
    instances = instances[keep]

    v = Visualizer(frame[:, :, ::-1],
                   MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(instances.to("cpu"))
    frame_with_boxes = v.get_image()[:, :, ::-1]

    if len(instances) > 0:

        is_duplicate = False

        for prev_img in prev_images:
            if compare_images(frame, prev_img):
                print("Duplicate image detected!",
                      compare_images(frame, prev_img))
                is_duplicate = True
                break

        if not is_duplicate:
            # Capture an image
            # Draw the predicted boxes on the frame

            uuid_var = str(uuid.uuid4())
            path = '/Users/pruthvipatel/Desktop/StrandAid_Object_Detection/images'
            filename = f'person_image_{uuid_var}.jpg'

            # saving the image in local storage
            cv2.imwrite(os.path.join(path, filename), frame_with_boxes)
            cv2.waitKey(10000)

        # Convert the image to bytes for storing in firebase
            converted_img = convertToJPG(frame_with_boxes)

        # Getting the labels of the detected objects
            labels = [MetadataCatalog.get(
                cfg.DATASETS.TRAIN[0]).thing_classes[class_id] for class_id in instances.pred_classes]
            print("Detected Labels:", labels[0])

        # Save the image to Firebase Storage
            try:

                print("Uploading image to Firebase Storage...")

                uploaded_image_url = upload_to_firebase(
                    filename, converted_img, labels[0])

            except Exception as e:

                print("Error uploading image to Firebase Storage:", e)

            # Add the new image to the previous images array
            prev_images.append(frame)

            if len(prev_images) > 4:
                prev_images.pop(0)

    cv2.imshow('frame', frame_with_boxes)

    key = cv2.waitKey(200)
    print(key)
    if key in [ord('a'), 1048673]:
        print('a pressed!')
    elif key in [27, 1048603]:
        # ESC key to abort, close window
        cv2.destroyAllWindows()
        break

    if cv2.waitKey(1) == ord('q'):
        break

# Release webcam and close all windows
cap.release()
cv2.destroyAllWindows()
