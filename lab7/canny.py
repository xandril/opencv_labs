import cv2


class CannyViewModel:
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

        self._min_contour_area = 30
        self._min_contour_vertices_count = 8
        self._max_contour_vertices_count = 30

    def run(self):
        cv2.imshow(self._img_name, self._img)

        cv2.createTrackbar('lower_canny_thr', self._img_name, 0, 255, self.left_canny_callback)
        cv2.createTrackbar('upper_canny_thr', self._img_name, 0, 255, self.right_canny_callback)
        cv2.createTrackbar('min_contour_vertices_count_thr', self._img_name, 0, 255,
                           self.min_contour_vertices_count_callback)
        cv2.createTrackbar('max_contour_vertices_count_thr', self._img_name, 0, 255,
                           self.max_contour_vertices_count_callback)
        cv2.createTrackbar('min contour are', self._img_name, 0, 255,
                           self.min_contour_area_callback)
        cv2.waitKey(0)

    def min_contour_area_callback(self, pos: int) -> None:
        self._min_contour_area = pos
        self._apply_canny(left_thr=self._left_pos, right_thr=self._right_pos, img=self._img)

    def min_contour_vertices_count_callback(self, pos: int) -> None:
        self._min_contour_vertices_count = pos
        self._apply_canny(left_thr=self._left_pos, right_thr=self._right_pos, img=self._img)

    def max_contour_vertices_count_callback(self, pos: int) -> None:
        self._max_contour_vertices_count = pos
        self._apply_canny(left_thr=self._left_pos, right_thr=self._right_pos, img=self._img)

    def left_canny_callback(self, pos: int) -> None:
        self._left_pos = pos
        self._apply_canny(left_thr=self._left_pos, right_thr=self._right_pos, img=self._img)

    def right_canny_callback(self, pos: int) -> None:
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

        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        print('contour number: ', len(contours))
        circle_contour_list = []
        circle_contour_area = []
        for contour in contours:
            area = cv2.contourArea(contour)
            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
            if (self._max_contour_vertices_count > len(approx) > self._min_contour_vertices_count) and (
                    area > self._min_contour_area):
                circle_contour_list.append(contour)
                circle_contour_area.append(area)

        print('final contour number: ', len(circle_contour_list))

        img_area = fig_img.shape[0] * fig_img.shape[1]
        for contour in circle_contour_list:
            x, y, w, h = cv2.boundingRect(contour)
            bb_area = w * h

            area_ratio = 100 * bb_area / img_area
            print('area_ratio: ', area_ratio)
            if area_ratio > 75:
                size = 'big'
            elif 50 < area_ratio < 75:
                size = 'medium'
            elif 25 < area_ratio < 50:
                size = 'small'
            else:
                size = 'very small'

            text = f'ellipse,{size}'
            cv2.putText(fig_img, text, (x + w // 4, y + h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1,
                        cv2.LINE_AA)
            cv2.drawContours(fig_img, [contour], 0, (0, 255, 0), 1)

        self._fig_image = fig_img
        cv2.imshow(self._fig_name, self._fig_image)

    def clear(self):
        cv2.destroyAllWindows()


if __name__ == '__main__':
    img_path = '../imgs/Venus_from_Mariner_10.jpg'

    canny_view_model = CannyViewModel(img_path)
    canny_view_model.run()
    canny_view_model.clear()
