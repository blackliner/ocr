import cv2
import imutils
import numpy as np
import pytesseract
from PIL import Image
import pafy


def on_trackbar(val):
    pass


epsilon = 0.05
min_area = 200
max_area = 10000

min_ar = 1.5
max_ar = 3.0


def get_number(img):
    img = cv2.resize(img, (1920, 1080))
    width = 1000
    left = int(1920 / 2 - width / 2)
    right = int(1920 / 2 + width / 2)

    img = img[600:900, left:right]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grey scale
    gray = cv2.bilateralFilter(gray, 11, 17, 17)  # Blur to reduce noise
    edged = cv2.Canny(gray, 30, 200)  # Perform Edge detection

    # find contours in the edged image, keep only the largest
    # ones, and initialize our screen contour
    cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:20]
    cnts = list(filter(lambda x: min_area < cv2.contourArea(x) < max_area, cnts))
    # cnts = list(filter(lambda x: 100 < x.size < 1000, cnts))

    # cv2.drawContours(img, cnts, -1, (0, 255, 0), 2)

    screenCnt = []
    text = []

    # loop over our contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon * peri, True)

        # if our approximated contour has four points, then
        # we can assume that we have found our screen
        if len(approx) == 4:

            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)

            if min_ar < ar < max_ar:
                screenCnt.append(approx)
                # break

    screenCnt = sorted(screenCnt, key=cv2.contourArea, reverse=True)
    cv2.drawContours(img, screenCnt, -1, (255, 0, 0), 3)

    for contour in screenCnt:

        # Masking the part other than the number plate
        mask = np.zeros(gray.shape, np.uint8)
        new_image = cv2.drawContours(mask, [contour], 0, 255, -1,)
        new_image = cv2.bitwise_and(img, img, mask=mask)

        # Now crop
        (x, y) = np.where(mask == 255)
        (topx, topy) = (np.min(x), np.min(y))
        (bottomx, bottomy) = (np.max(x), np.max(y))
        Cropped = gray[topx : bottomx + 1, topy : bottomy + 1]

        # Read the number plate
        text.append(pytesseract.image_to_string(Cropped))

    return text, img, gray, edged


url = "https://www.youtube.com/watch?v=21ftjuie9CU"
url = "https://www.youtube.com/watch?v=8RL4Q_GN8YQ"
# url = "https://www.youtube.com/watch?v=50kr23cMUBc"
vPafy = pafy.new(url)
print(vPafy)
play = vPafy.getbest()
print(play)

# start the video
cap = cv2.VideoCapture(play.url)


cv2.namedWindow("image", flags=cv2.WINDOW_AUTOSIZE)
cv2.createTrackbar("epsilon", "image", int(epsilon * 1000), 100, on_trackbar)
cv2.createTrackbar("min_area", "image", min_area, 10000, on_trackbar)
cv2.createTrackbar("max_area", "image", max_area, 10000, on_trackbar)
cv2.createTrackbar("min_ar", "image", int(10 * min_ar), 100, on_trackbar)
cv2.createTrackbar("max_ar", "image", int(10 * max_ar), 100, on_trackbar)
pause = False

while True:
    if not pause:
        ret, video = cap.read()

    # img = cv2.imread("ocr/4.jpg", cv2.IMREAD_COLOR)
    # print(img)

    epsilon = cv2.getTrackbarPos("epsilon", "image") / 1000
    min_area = cv2.getTrackbarPos("min_area", "image")
    max_area = cv2.getTrackbarPos("max_area", "image")
    min_ar = cv2.getTrackbarPos("min_ar", "image") / 10
    max_ar = cv2.getTrackbarPos("max_ar", "image") / 10

    text, img, gray, edged = get_number(video)

    if text:
        print("Detected Number is:", text)

    cv2.imshow("image", img)
    # cv2.imshow("gray", gray)
    # cv2.imshow("edged", edged)
    # cv2.imshow("Cropped", Cropped)

    key = cv2.waitKey(16)
    # print(key)
    if key == ord("q"):
        break
    elif key == ord("k"):
        pause = not pause
    elif key == ord("l"):
        cpos = cap.get(cv2.CAP_PROP_POS_MSEC)
        cap.set(cv2.CAP_PROP_POS_MSEC, cpos + 5000)
        ret, video = cap.read()
    elif key == ord("j"):
        cpos = cap.get(cv2.CAP_PROP_POS_MSEC)
        cap.set(cv2.CAP_PROP_POS_MSEC, cpos - 5000)
        ret, video = cap.read()


cap.release()
cv2.destroyAllWindows()
