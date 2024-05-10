import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None, "Image not found"

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.add(mask1, mask2)

    target_red = cv2.bitwise_and(gray, gray, mask=red_mask)
    blurred = cv2.GaussianBlur(target_red, (5, 5), 1.2)
    edges = cv2.Canny(blurred, 50, 150)

    return edges, image

def detect_prohibition_sign(edges, original_image):
    height, width = edges.shape
    dp = 1.2
    minDist = int(height * 0.05)
    param1 = 70
    param2 = 30
    minRadius = int(height * 0.02)
    maxRadius = int(height * 0.1)

    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, dp, minDist,
                               param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(original_image, (i[0], i[1]), i[2], (0, 255, 0), 2)
        return original_image
    else:
        return "No prohibition signs detected."

def main():
    # Specifying the range from xd1 to xd12
    image_titles = [f'/Users/nikip/Desktop/xd{i}.jpg' for i in range(1, 13)]
    results = []

    for title in image_titles:
        edges, original_image = preprocess_image(title)
        if edges is None:
            results.append("Image not found")  # Use a placeholder if the image is not found
        else:
            result = detect_prohibition_sign(edges, original_image)
            results.append(result)

    # Plotting results in three separate figures without titles, each with four images
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

main()