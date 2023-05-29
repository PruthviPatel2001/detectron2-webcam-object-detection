import cv2


def convertToJPG(image):
    # Convert the image to bytes for storing in firebase
    try:
        _, image_encoded = cv2.imencode('.jpg', image)
        image_bytes = image_encoded.tobytes()
        return image_bytes
    except Exception as e:
        print("Error converting image to bytes:", e)
        return None
