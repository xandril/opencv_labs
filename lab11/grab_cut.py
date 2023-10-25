import cv2
import numpy as np

if __name__ == '__main__':
    # Load the image
    img = cv2.imread('../imgs/img42.jpg')

    # Create a mask image
    mask = np.zeros(img.shape[:2], np.uint8)

    # Define the region of interest (ROI) using a rectangle
    rect = cv2.selectROI(windowName='select roi', img=img, showCrosshair=False)

    # Set up the initial state of the grabCut algorithm
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # Run the grabCut algorithm with mask mode
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    # Create a new mask image with only the foreground pixels
    new_mask = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')

    # Apply the new mask to the original image
    result = cv2.bitwise_and(img, img, mask=new_mask)

    # Display the result
    cv2.imshow('Result', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
