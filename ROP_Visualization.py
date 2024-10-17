#Step 8: Visualization of ROP: Generate target bounding boxes and draw lesion contours
import cv2
import matplotlib.pyplot as plt

# read the origial image and mask
img = cv2.imread('dataset/img_mask/images/48.png')
#ground_truth = cv2.imread('dataset/img_mask/masks/48.png', cv2.IMREAD_GRAYSCALE)
predition = cv2.imread('dataset/predit/output.png', cv2.IMREAD_GRAYSCALE)

# Extract the contour of the mask:
# cv2.RETR_EXTERNAL: Only retrieve the highest level (outermost) contour.
# cv2.CHAIN_APPROX_SIMPLE: Compress the redundant points in the vertical, horizontal and diagonal directions, and only keep the key points.
contours, _ = cv2.findContours(predition, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Extract target detection box
x, y, w, h = cv2.boundingRect(predition)

# Draw the contours on the original image
img_contours = img.copy()
cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 1)

# Draw the target detection box on the original image
img_bbox = img.copy()
cv2.rectangle(img_bbox, (x, y), (x + w, y + h), (255, 0, 0), 1)

# show result
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img_contours, cv2.COLOR_BGR2RGB))
plt.title('ROP Contours')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(img_bbox, cv2.COLOR_BGR2RGB))
plt.title('ROP Bounding Box')

plt.show()

