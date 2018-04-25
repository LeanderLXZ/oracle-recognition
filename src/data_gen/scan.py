from skimage import filters
import numpy as np
import cv2
import imutils


class Scan(object):

  def __init__(self):
    pass

  @staticmethod
  def _detect_edge(image, show_img=False):
    print("STEP 1: Edge Detection")
    # convert the image to gray scale, blur it, and find edges
    # in the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edge = cv2.Canny(gray, 75, 200)

    # show the original image and the edge detected image
    if show_img:
      cv2.imshow("Edged", imutils.resize(edge, height=650))
    return edge

  @staticmethod
  def _detect_contours(image, edge, save_img=False, show_img=False):

    print("STEP 2: Find contours of paper")
    # find the contours in the edged image, keeping only the
    # largest ones, and initialize the screen contour
    cnts = cv2.findContours(edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    # loop over the contours
    screen_cnt = None
    for c in cnts:
      # approximate the contour
      peri = cv2.arcLength(c, True)
      approx = cv2.approxPolyDP(c, 0.02 * peri, True)

      # if our approximated contour has four points, then we
      # can assume that we have found our screen
      if len(approx) == 4:
        screen_cnt = approx
        break

    # show the contour (outline) of the piece of paper
    cv2.drawContours(image, [screen_cnt], -1, (0, 255, 0), 2)
    if save_img:
      cv2.imwrite("../data/outline.jpg", image)
    if show_img:
      cv2.imshow("Outline", imutils.resize(image, height=650))

    return screen_cnt

  def _transform(self, image, contours,
                 ratio=1., save_img=False, show_img=False):

    print("STEP 3: Apply perspective transform")
    # apply the four point transform to obtain a top-down
    # view of the original image
    warped = self._four_point_transform(image, contours.reshape(4, 2) * ratio)

    # convert the warped image to gray scale, then threshold it
    # to give it that 'black and white' paper effect
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    t = filters.threshold_local(warped, 11, offset=10, method="gaussian")
    warped = np.array((warped > t)).astype("uint8") * 255

    # show the original and scanned images
    if save_img:
      cv2.imwrite("../data/warped.jpg", warped)
    if show_img:
      cv2.imshow("Scanned", imutils.resize(warped, height=650))

  @staticmethod
  def _order_points(pts):

    # Initialize a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # The top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # Now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # Return the ordered coordinates
    return rect

  def _four_point_transform(self, image, pts):

    # Obtain a consistent order of the points and unpack them
    # individually
    rect = self._order_points(pts)
    (tl, tr, br, bl) = rect

    # Compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordinates or the top-right and top-left x-coordinates
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    # Now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
      [0, 0],
      [max_width - 1, 0],
      [max_width - 1, max_height - 1],
      [0, max_height - 1]], dtype="float32")

    # Compute the perspective transform matrix and then apply it
    m = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, m, (max_width, max_height))

    # Return the warped image
    return warped

  def scan(self, image_path, show_img=False):
    # load the image and compute the ratio of the old height
    # to the new height, clone it, and resize it
    _image = cv2.imread(image_path)
    if show_img:
      cv2.imshow("Origin", imutils.resize(_image, height=650))
    _edge = self._detect_edge(_image, show_img=show_img)
    _contours = self._detect_contours(_image, _edge, show_img=show_img)
    self._transform(_image, _contours, show_img=show_img)
    if show_img:
      cv2.waitKey(0)


if __name__ == '__main__':
  S = Scan()
  S.scan('../data/scan/new/扫描件0409105438_1_1005.jpg', show_img=True)
