from datetime import timedelta
import cv2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.model_zoo import model_zoo
import uuid
import os
from PIL import Image
import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage


cred = credentials.Certificate(
    "/Users/pruthvipatel/Desktop/StrandAid_Object_Detection/strandaid-16e48-firebase-adminsdk-1dltc-e05e292a85.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'strandaid-16e48.appspot.com'
})


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

while True:
    ret, frame = cap.read()

    # Run object detection on the frame
    outputs = predictor(frame)

    # Filter out all other classes except for "person"
    instances = outputs["instances"]
    keep = (instances.pred_classes == 0) | (instances.pred_classes == 16)
    instances = instances[keep]

    # Draw the predicted boxes on the frame
    v = Visualizer(frame[:, :, ::-1],
                   MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    v = v.draw_instance_predictions(instances.to("cpu"))
    frame = v.get_image()[:, :, ::-1]

    if len(instances) > 0:
        # Capture an image
        uuid_var = str(uuid.uuid4())
        path = '/Users/pruthvipatel/Desktop/StrandAid_Object_Detection/images'
        filename = f'person_image_{uuid_var}.jpg'
        cv2.imwrite(os.path.join(path, filename), frame)
        cv2.waitKey(7000)
        _, image_encoded = cv2.imencode('.jpg', frame)
        image_bytes = image_encoded.tobytes()

        # Save the image to Firebase Storage
        try:
            bucket = storage.bucket()
            blob = bucket.blob(filename)
            blob.upload_from_string(image_bytes, content_type='image/jpeg')
            print("Image uploaded to Firebase Storage successfully!")

            url = blob.generate_signed_url(
                expiration=timedelta(days=7), method='GET')
            print("Image URL:", url)
        except Exception as e:
            print("Error uploading image to Firebase Storage:", e)

    cv2.imshow('frame', frame)

    key = cv2.waitKey(200)
    print(key)
    if key in [ord('a'), 1048673]:
        print('a pressed!')
    elif key in [27, 1048603]:  # ESC key to abort, close window
        cv2.destroyAllWindows()
        break

    if cv2.waitKey(1) == ord('q'):
        break

# Release webcam and close all windows
cap.release()
cv2.destroyAllWindows()
