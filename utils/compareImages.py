import cv2
from skimage.metrics import structural_similarity as ssim


def compare_images(img1, img2):
    # Convert images to grayscale
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Calculate the structural similarity index (SSIM)
    similarity = ssim(img1_gray, img2_gray)

    print("Similarity:", similarity)

    # Return True if the images are similar (SSIM close to 1), False otherwise
    return similarity > 0.79  # Adjust the threshold as needed
