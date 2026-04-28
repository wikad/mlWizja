import cv2
import numpy as np

class Panel:
    """Reprezentuje pojedynczy wykryty panel i jego geometrię."""
    def __init__(self, contour, target_size=(200, 100)):
        self.contour = contour
        self.target_size = target_size
        self.rect_raw = cv2.minAreaRect(contour)
        self.box = self._order_points(cv2.boxPoints(self.rect_raw))
        self.width, self.height = self.rect_raw[1]
        self.angle = self._calculate_angle()
        
        # Macierze transformacji
        self.M = cv2.getPerspectiveTransform(self.box, np.array([
            [0, 0], [target_size[0], 0], 
            [target_size[0], target_size[1]], [0, target_size[1]]
        ], dtype="float32"))
        self.M_inv = np.linalg.inv(self.M)

    def __str__(self):
        """To wywoła print()."""
        return f"Panel(angle={self.angle:.1f}, size={int(self.width)}x{int(self.height)})"

    def __repr__(self):
        """To wywoła print(lista_obiektów)."""
        return self.__str__()

    def _order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
        return rect

    def _calculate_angle(self):
        p1, p2 = self.box[0], self.box[1]
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        raw_deg = np.degrees(np.arctan2(dy, dx))
        angle = raw_deg + 90 if self.width > self.height else raw_deg
        return (angle + 90) % 180 - 90

    def extract_warped(self, frame):
        """Wycina i prostuje panel z oryginalnej klatki."""
        warped = cv2.warpPerspective(frame, self.M, self.target_size)
        return cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    
    def extract_warped_bgr(self, frame):
        """Wycina i prostuje panel w kolorze."""
        return cv2.warpPerspective(frame, self.M, self.target_size)
    
    def get_cells(self, warped_img, rows=10, cols=20):
        """Tnie wyprostowany panel na siatkę komórek z wewnętrznym marginesem."""
        h, w = warped_img.shape[:2]
        cell_h, cell_w = h // rows, w // cols
        
        for r in range(rows):
            for c in range(cols):
                # 1. Wycinamy pełną komórkę
                y1, y2 = r * cell_h, (r + 1) * cell_h
                x1, x2 = c * cell_w, (c + 1) * cell_w
                cell_img = warped_img[y1:y2, x1:x2]
                
                # 2. Obliczamy marginesy (np. 10% z każdej strony)
                margin_h = max(1, int(cell_h * 0.1)) # min. 1 piksel, by nie było błędu
                margin_w = max(1, int(cell_w * 0.1))
                
                # 3. Wycinamy środek komórki
                # Używamy slice [od:do, od:do]
                cell_img_inner = cell_img[margin_h:-margin_h, margin_w:-margin_w]

                # 4. Zwracamy czysty środek, ale kordy (x1, y1) zostawiamy oryginalne
                # dzięki temu Visualizer będzie wiedział, gdzie narysować ramkę na pełnym panelu
                yield cell_img_inner, r, c, (x1, y1, cell_w, cell_h)