import cv2


class CannyViewModel:
    def __init__(self, img_path: str):
        self._img = cv2.imread(img_path)
        self._img_name = "Image"

        self._right_pos = 0
        self._left_pos = 0

        self._edges: cv2.typing.MatLike = None
        self._edges_name = "Edges"

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

        self._edges = edges
        cv2.imshow(self._edges_name, self._edges)

    def clear(self):
        cv2.destroyAllWindows()


if __name__ == '__main__':
    img_path = '../imgs/Venus_from_Mariner_10.jpg'

    canny_view_model = CannyViewModel(img_path)
    canny_view_model.run()
    canny_view_model.clear()
