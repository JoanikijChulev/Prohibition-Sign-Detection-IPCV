import cv2
import numpy as np
import matplotlib.pyplot as plt

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

def modified_hough_transform(image):
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp=1.2, minDist=170,
                               param1=70, param2=30, minRadius=10, maxRadius=300)
    return circles

def draw_detected_circles(original_image, circles):
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(original_image, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv2.circle(original_image, (i[0], i[1]), 2, (0, 0, 255), 3)

def process_image(image_path):
    img = load_image(image_path)
    if img is None:
        return None, "Error loading image"

    hsv_img = convert_to_hsv(img)
    thresholded_img = apply_hsv_threshold(hsv_img)
    denoised_img = denoise_image(thresholded_img)
    circles = modified_hough_transform(denoised_img)
    if circles is not None:
        draw_detected_circles(img, circles)

    return img, circles

def main():
    image_paths = [f'/Users/nikip/Desktop/xd{i}.jpg' for i in range(1, 13)]
    results = []

    for path in image_paths:
        img, circles = process_image(path)
        results.append(img if img is not None else "Error loading image")

    # Display results in three separate figures, each with four images
    for i in range(3):  # Loop to create three figures
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))  # 2 columns, 2 rows
        for j, ax in enumerate(axs.flatten()):
            idx = i * 4 + j
            if idx < len(results):
                img = results[idx]
                if isinstance(img, str):
                    ax.text(0.5, 0.5, img, ha='center', va='center', transform=ax.transAxes)
                else:
                    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.axis('off')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()