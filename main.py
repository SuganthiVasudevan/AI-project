import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def load_and_preprocess(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply histogram equalization
    gray = cv2.equalizeHist(gray)
    return gray

def align_images(before_img, after_img):
    # Using SIFT for feature detection and matching
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(before_img, None)
    keypoints2, descriptors2 = sift.detectAndCompute(after_img, None)

    if descriptors1 is None or descriptors2 is None:
        raise ValueError("No descriptors found in one of the images.")

    # Using FLANN based matcher for better performance
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 4:
        raise ValueError("Not enough good matches found between the images.")

    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    if src_pts.shape[0] < 4 or dst_pts.shape[0] < 4:
        raise ValueError("Not enough keypoints after reshaping.")

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if M is None:
        raise ValueError("Homography matrix could not be computed.")

    aligned_after_img = cv2.warpPerspective(after_img, M, (before_img.shape[1], before_img.shape[0]))

    return aligned_after_img

def calculate_damage(before_img, after_img):
    # Calculate Structural Similarity Index (SSIM) and difference image
    score, diff = ssim(before_img, after_img, full=True)
    diff = (diff * 255).astype("uint8")

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Apply morphological operations to clean up the binary image
    kernel = np.ones((5,5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Calculate damage percentage
    damage_percentage = (np.sum(thresh == 255) / thresh.size) * 100
    return damage_percentage

before_image_path = r"C:\Users\Suganthi V\OneDrive\Documents\before image.jpg"
after_image_path = r"C:\Users\Suganthi V\OneDrive\Documents\after image.jpg"

before_img = load_and_preprocess(before_image_path)
after_img = load_and_preprocess(after_image_path)

aligned_after_img = align_images(before_img, after_img)

damage_percentage = calculate_damage(before_img, aligned_after_img)

print(f"Damage Percentage: {damage_percentage:.2f}%")
