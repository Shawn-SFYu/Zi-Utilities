"""
For test purpose only
Visualize different imgs processing methods: denoise, upsample, threshold
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Read image
img = cv2.imread("test0.png")

fig, axs = plt.subplots(4, 2)

axs[0, 0].imshow(img)
axs[0, 0].set_title('Original')


""" Denoise 1"""
denoised = cv2.bilateralFilter(img, 9, 75, 75)
axs[0, 1].imshow(denoised)
axs[0, 1].set_title('denoised')


""" Denoise 2"""
denoised2 = cv2.bilateralFilter(img, 13, 100, 100)
axs[1, 0].imshow(denoised2)
axs[1, 0].set_title('denoised2')


"""Super resolution"""
esdr = cv2.dnn_superres.DnnSuperResImpl_create()
# Set the parameters for the ESDR algorithm
model_path = "EDSR_x4.pb"
esdr.readModel(model_path)
esdr.setModel("edsr", 4)
# Upscale the image
img_hr = esdr.upsample(denoised2)
axs[1, 1].imshow(img_hr)
axs[1, 1].set_title('ESDR')

"""Basic parameters after super-reso"""
height, width = img_hr.shape[:2]
center = np.array([width // 2, height // 2])


"""Thresholding"""
# this may not be desired due to the jigsaw effect
gray_img = cv2.cvtColor(img_hr, cv2.COLOR_RGB2GRAY)
thresh_value, thresh_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

"""Img mode detection light/dark"""
dark_distance = 0
light_distance = 0
dark_count = 0
light_count = 0
for x in range(width):
    for y in range(height):
        if thresh_img[y, x] == 0:
            dark_distance += np.linalg.norm(center - np.array([y, x]))
            dark_count += 1
        else:
            light_distance += np.linalg.norm(center - np.array([y, x]))
            light_count += 1
dark_distance = dark_distance / dark_count
light_distance = light_distance / light_count
light_mode = True if light_distance < dark_distance else False
print(f"Light mode is {light_mode}")
if light_mode == False:
    thresh_img = cv2.bitwise_not(thresh_img)
    gray_img = cv2.bitwise_not(gray_img)

axs[2, 0].imshow(thresh_img)
axs[2, 0].set_title('Mask')

kernel = np.ones((5, 5), np.uint8)
dilated_mask = cv2.dilate(thresh_img, kernel, iterations=1)
axs[2, 1].imshow(dilated_mask)
axs[2, 1].set_title('diluted_mask')

"""Scale to dimension"""
masked_img = cv2.bitwise_and(gray_img, gray_img, mask=dilated_mask)
desired_resolution = 512

max_dim = max(height, width)
square_img = np.zeros((max_dim, max_dim), dtype=np.uint8)
x_offset = (max_dim - width) // 2
y_offset = (max_dim - height) // 2
square_img[y_offset:y_offset+height, x_offset:x_offset+width] = masked_img
resized_img = cv2.resize(square_img, (desired_resolution, desired_resolution),\
                         interpolation=cv2.INTER_LINEAR)
axs[3, 0].imshow(resized_img)
axs[3, 0].set_title('denoised')


""""""




"""
denoised3 = cv2.bilateralFilter(img, 15, 150, 150)
axs[2, 0].imshow(denoised3)
axs[2, 0].set_title('denoised')

denoised4 = cv2.bilateralFilter(img, 19, 175, 175)
axs[2, 1].imshow(denoised4)
axs[2, 1].set_title('denoised')
"""

# Add a title for the entire figure
fig.suptitle('My 2x2 Plot')
plt.show()
