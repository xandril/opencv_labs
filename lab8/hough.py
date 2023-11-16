import cv2 as cv
import numpy as np

RHO = 1
ANGLE = np.pi / 180
THR = 100
MIN_LINE_LENGTH = 50
MAX_LINE_GAP = 10

if __name__ == "__main__":
    filename = '../imgs/road.png'
    # Loads an image
    src = cv.imread(filename, cv.IMREAD_GRAYSCALE)
    # Check if image is loaded fine
    if src is None:
        raise ValueError('Error opening image!')

    dst = cv.Canny(src, 50, 200, None, 3)

    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)
    cdstP = np.copy(cdst)

    linesP = cv.HoughLinesP(dst, RHO, ANGLE, THR, None, MIN_LINE_LENGTH, MAX_LINE_GAP)

    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv.LINE_AA)

    cv.imshow("Source", src)
    cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)

    cv.waitKey()
