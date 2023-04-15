import cv2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.model_zoo import model_zoo

from PIL import Image

# Load config and model
cfg = get_cfg()
cfg.MODEL.DEVICE = 'cpu'  # Set device to CPU
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = '/Users/pruthvipatel/Desktop/objectdetection/model_final_f10217.pkl'
predictor = DefaultPredictor(cfg)

# Set up webcam
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()

    # Run object detection on the frame
    outputs = predictor(frame)

    # TODO: Post-process the outputs and draw the predicted boxes on the frame
    # You can refer to the Detectron2 documentation for more information on how to do this

    # Visualize the output
    v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    frame = v.get_image()[:, :, ::-1]

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Release webcam and close all windows
cap.release()
cv2.destroyAllWindows()
