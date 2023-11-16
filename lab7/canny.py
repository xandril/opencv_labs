import cv2
import numpy as np


class CannyView:
    def __init__(self, img_path: str):
        self._img = cv2.imread(img_path)
        self._img_name = "Image"

        self._right_pos = 0
        self._left_pos = 0

        self._edges: cv2.typing.MatLike = None
        self._edges_name = "Edges"

        self._contours_image: cv2.typing.MatLike = None
        self._contours_name = 'Contours results'

        self._fig_image: cv2.typing.MatLike = None
        self._fig_name = "figures"

    def run(self):
        cv2.imshow(self._img_name, self._img)

        cv2.createTrackbar('lower_thr', self._img_name, 0, 255, self.left_callback)
        cv2.createTrackbar('upper_thr', self._img_name, 0, 255, self.right_callback)
        cv2.waitKey(0)

    def left_callback(self, pos: int) -> None:
        self._left_pos = pos
        self._apply_canny(left_thr=self._left_pos, right_thr=self._right_pos, img=self._img)

    def right_callback(self, pos: int) -> None:
        self._right_pos = pos
        self._apply_canny(left_thr=self._left_pos, right_thr=self._right_pos, img=self._img)

    def _apply_canny(self, left_thr: int, right_thr: int, img: cv2.typing.MatLike):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.blur(gray, (5, 5))

        # Apply Canny edge detection
        edges = cv2.Canny(gray, left_thr, right_thr)

        # Find contours
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        self._contours = contours
        result = img.copy()
        # Draw contours on original image
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

        self._edges = edges
        self._contours_image = result

        cv2.imshow(self._edges_name, self._edges)
        cv2.imshow(self._contours_name, self._contours_image)
        self._approx_contours()

    def _approx_contours(self):
        fig_img = self._img.copy()
        edges = self._edges.copy()
        # dilated = edges

        # cv2.imshow('delated', dilated)

        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        print('counter number: ', len(contours))

        ref_triangle = np.array([[0, 0], [0, 50], [50, 50]], dtype=np.int32)
        ref_rectangle = np.array([[0, 0], [0, 50], [50, 50], [50, 0]], dtype=np.int32)
        ref_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))

        # Compute Hu Moments of reference shapes
        moments_triangle = cv2.HuMoments(cv2.moments(ref_triangle)).flatten()
        moments_rectangle = cv2.HuMoments(cv2.moments(ref_rectangle)).flatten()
        moments_ellipse = cv2.HuMoments(cv2.moments(ref_ellipse)).flatten()

        for contour in contours:
            # Approximate the contour
            moments_cnt = cv2.HuMoments(cv2.moments(contour)).flatten()

            # Compute shape similarity with reference shapes
            similarity_triangle = cv2.matchShapes(moments_cnt, moments_triangle, cv2.CONTOURS_MATCH_I2, 0)
            similarity_rectangle = cv2.matchShapes(moments_cnt, moments_rectangle, cv2.CONTOURS_MATCH_I2, 0)
            similarity_ellipse = cv2.matchShapes(moments_cnt, moments_ellipse, cv2.CONTOURS_MATCH_I2, 0)

            x, y, w, h = cv2.boundingRect(contour)
            text = 'idk'
            sim = 0.01
            res_sim = 42
            # Check if contour is a triangle
            if similarity_triangle < sim:
                text = 'triangle'
                res_sim = similarity_triangle
            # Check if contour is a rectangle
            elif similarity_rectangle < sim:
                text = 'rectangle'
                res_sim = similarity_rectangle

            # Check if contour is an ellipse
            elif similarity_ellipse < sim:
                text = 'ellipse'
                res_sim = similarity_ellipse
            text += str(res_sim)
            cv2.drawContours(fig_img, [contour], 0, (0, 255, 0), 1)
            cv2.putText(fig_img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

        self._fig_image = fig_img
        cv2.imshow(self._fig_name, self._fig_image)

    def clear(self):
        cv2.destroyAllWindows()


if __name__ == '__main__':
    img_path = '../imgs/triangle.jpg'

    canny_view = CannyView(img_path)
    canny_view.run()
    canny_view.clear()
