
import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Read the image
image = cv2.imread('/content/harris_corner_detection_image.jpg')

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Convert image to float
gray = np.float32(gray)

# Apply Harris Corner Detection
corners = cv2.cornerHarris(gray, 2, 3, 0.04)

# Dilate the corners
corners = cv2.dilate(corners, None)

# Draw red circles on detected corners
result = image.copy()

threshold = 0.01 * corners.max()

for y in range(0, image.shape[0], 3):
    for x in range(0, image.shape[1], 3):
        if corners[y, x] > threshold:
            cv2.circle(result, (x, y), 2, (0, 0, 255), 1)

# Save the result
cv2.imwrite('/content/image_after.png', result)

# Display the result
cv2_imshow(result)
