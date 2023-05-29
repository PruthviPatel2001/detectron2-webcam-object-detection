import cv2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.model_zoo import model_zoo
import uuid
import os

from utils.jpgConverter import convertToJPG

from firebase_files.index import upload_to_firebase


# Load config and model
cfg = get_cfg()
cfg.MODEL.DEVICE = 'cpu'  # Set device to CPU
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = '/Users/pruthvipatel/Desktop/objectdetection/model_final_f10217.pkl'
predictor = DefaultPredictor(cfg)

# Path to the input image
image_path = '/Users/pruthvipatel/Desktop/StrandAid_Object_Detection/test_images/person_11.jpeg'

# Read the input image
frame = cv2.imread(image_path)

# Run object detection on the image
outputs = predictor(frame)

# Filter out all other classes except for "person"
instances = outputs["instances"]
keep = (instances.pred_classes == 0) | (instances.pred_classes == 16)
instances = instances[keep]

# Draw the predicted boxes on the image
v = Visualizer(frame[:, :, ::-1],
               MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
v = v.draw_instance_predictions(instances.to("cpu"))
output_image = v.get_image()[:, :, ::-1]

if len(instances) > 0:
    # Capture an image
    uuid_var = str(uuid.uuid4())
    output_path = '/Users/pruthvipatel/Desktop/StrandAid_Object_Detection/images/test_image_result'
    filename = f'person_image_{uuid_var}.jpg'

    # Saving the output image with detected objects
    cv2.imwrite(os.path.join(output_path, filename), output_image)

    # Convert the output image to bytes for storing in Firebase
    converted_img = convertToJPG(output_image)

    # Getting the labels of the detected objects
    labels = [MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes[class_id]
              for class_id in instances.pred_classes]
    print("Detected Labels:", labels[0])


# Display the output image with detected objects
cv2.imshow('output', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
