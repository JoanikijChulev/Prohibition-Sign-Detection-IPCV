import cv2
import numpy as np
from sklearn.model_selection import ParameterGrid

def load_image(image_path):
    return cv2.imread(image_path)

def convert_to_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def apply_hsv_threshold(hsv_image):
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    full_mask = cv2.bitwise_or(mask1, mask2)
    return full_mask

def denoise_image(image):
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return closing

def modified_hough_transform(image, dp, minDist, param1, param2, minRadius, maxRadius):
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist,
                               param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
    return circles

def draw_detected_circles(original_image, circles):
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(original_image, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(original_image, (i[0], i[1]), 2, (0, 0, 255), 3)

def calculate_score(original_image, hsv_image, circles):
    if circles is None or len(circles[0, :]) == 0:
        return 0

    # Initialize the score
    score = 0
    for circle in circles[0, :]:
        center_x, center_y, radius = circle
        if check_boundary(hsv_image, center_x, center_y, radius):
            score += 1
    
    # Penalty for multiple detections - the ideal is to detect exactly one circle per sign
    if len(circles[0, :]) > 1:
        score -= len(circles[0, :]) - 1

    return score

def check_boundary(hsv_image, center_x, center_y, radius):
    num_red_pixels = 0
    num_total_pixels = 0
    for angle in range(0, 360, 5):
        x = int(center_x + radius * np.cos(angle * np.pi / 180))
        y = int(center_y + radius * np.sin(angle * np.pi / 180))
        if 0 <= x < hsv_image.shape[1] and 0 <= y < hsv_image.shape[0]:
            pixel = hsv_image[y, x]
            if (0 <= pixel[0] <= 10 or 160 <= pixel[0] <= 180) and pixel[1] >= 100 and pixel[2] >= 100:
                num_red_pixels += 1
            num_total_pixels += 1
    # Require at least 50% of the pixels on the perimeter to be red
    return num_red_pixels > 0.5 * num_total_pixels

def main(image_path):
    img = load_image(image_path)
    if img is None:
        print("Error loading image.")
        return

    hsv_img = convert_to_hsv(img)
    thresholded_img = apply_hsv_threshold(hsv_img)
    denoised_img = denoise_image(thresholded_img)

    param_grid = {
        'dp': [1.0, 1.1, 1.15, 1.2, 1.5],
        'minDist': [100, 150, 200],
        'param1': [50, 100],
        'param2': [10, 20, 30, 40],
        'minRadius': [10, 20],
        'maxRadius': [100, 250]
    }
    best_score = -np.inf
    best_params = None

    for params in ParameterGrid(param_grid):
        circles = modified_hough_transform(denoised_img, **params)
        score = calculate_score(img, hsv_img, circles)
        if score > best_score:
            best_score = score
            best_params = params
            best_circles = circles

    if best_score > 0:
        print(f"Best Score: {best_score}, with parameters: {best_params}")
        draw_detected_circles(img, best_circles)
    else:
        print("No traffic signs detected with any parameter set.")

    # Resize the image before displaying
    scale_percent = 50  # percentage of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    cv2.imshow('Detected Traffic Signs', resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main('/Users/nikip/Desktop/xd12.jpg')
