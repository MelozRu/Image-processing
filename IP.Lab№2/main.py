import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def find_erosion_zones(img):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    image_erode = cv.erode(img, kernel)

    hsv_img = cv.cvtColor(image_erode, cv.COLOR_BGR2HSV)
    #выделяем зелёный
    healthy_part = cv.inRange(hsv_img, (36,25,30), (86, 255, 255))

    markers = np.zeros((img.shape[0], img.shape[1]), dtype="int32")
    markers[healthy_part>1] = 255
    markers[236:255, 0:20] = 1
    markers[0:20, 0:20] = 1
    markers[0:20, 236:255] = 1
    markers[236:255, 236:255] = 1

    leafs_area_BGR = cv.watershed(image_erode, markers)
    erosion_zone = leafs_area_BGR - healthy_part
    mask = np.zeros_like(img, np.uint8)
    mask [leafs_area_BGR > 1] = (255 , 0, 255)
    mask[erosion_zone > 1] = (0, 0, 255)
    return mask


def remove_shadows(img):
    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    b, g, r = cv.split(img[0:20, 235:255])
    b2, g2, r2 = cv.split(img[235:255, 0:20])
    b = np.append(b, b2, 1)
    g = np.append(g, g2, 1)
    r = np.append(r, r2, 1)
    black_pixels = np.where(
        (hsv_img[:, :, 2] < 45)
    )
    img[black_pixels] = [np.median(b), np.median(g), np.median(r)]
    return img

columns = 5
rows = 1
fig =  plt.figure(figsize=(10, 7))
img = cv.imread("7.jpg", cv.IMREAD_COLOR)

fig.add_subplot(rows, columns, 1)
result = find_erosion_zones(img)
plt.imshow(img)
plt.title("unfiltered")


wihout_shadows = remove_shadows(img)

fig.add_subplot(rows, columns, 2)
result = find_erosion_zones(img)
plt.imshow(img)
plt.title("wihout_shadows")

gauss = cv.GaussianBlur(wihout_shadows, (7,7), cv.BORDER_DEFAULT)
result = find_erosion_zones(gauss)
fig.add_subplot(rows, columns, 3)
plt.imshow(result)
plt.title("Gaussian")

bilateral = cv.bilateralFilter(wihout_shadows, 15, 75, 75)
result = find_erosion_zones(bilateral)
fig.add_subplot(rows, columns, 4)
plt.imshow(result)
plt.title("Bilateral")

non_local = cv.fastNlMeansDenoisingColored(img, None, 3, 3, 7, 21)
result = find_erosion_zones(non_local)
fig.add_subplot(rows, columns, 5)
plt.imshow(result)
plt.title("Non-Local Means")

plt.show()

