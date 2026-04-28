import cv2
import numpy as np
import os
from ultralytics import YOLO

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

class PanelDetector:
    """Obsługuje model YOLO i ekstrakcję konturów."""
    def __init__(self, model_path, conf=0.6):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect_panels(self, frame):
        results = self.model(frame, conf=self.conf, verbose=False)
        merged_contours = self._get_merged_contours(results, frame.shape)
        
        panels = []
        for cnt in merged_contours:
            if cv2.contourArea(cnt) > 1000:
                panels.append(Panel(cnt))
        return panels

    def _get_merged_contours(self, results, frame_shape):
        if not results[0].masks: return []
        mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        for m in results[0].masks.data:
            m_res = cv2.resize(m.cpu().numpy(), (frame_shape[1], frame_shape[0]))
            mask[m_res > 0.5] = 255
        
        kernel = np.ones((20, 20), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

class PanelAnalyzer:
    """Logika analizy zmian i siatki błędów."""
    def __init__(self, ref_dir="refs", grid_size=(10, 10), threshold=25):
        self.ref_dir = ref_dir
        self.grid_size = grid_size
        self.threshold = threshold
        if not os.path.exists(ref_dir): os.makedirs(ref_dir)

    def get_grid_stats(self, gray_panel):
        gh, gw = self.grid_size
        h, w = gray_panel.shape
        step_h, step_w = h // gh, w // gw
        
        # Wyrównanie histogramu dla stabilności
        norm = cv2.equalizeHist(gray_panel)
        
        grid = np.array([[
            np.mean(norm[i*step_h:(i+1)*step_h, j*step_w:(j+1)*step_w])
            for j in range(gw)] for i in range(gh)
        ])
        return grid

    def check_changes(self, panel_id, current_grid):
        ref_path = os.path.join(self.ref_dir, f"panel_{panel_id}.npy")
        
        if not os.path.exists(ref_path):
            np.save(ref_path, current_grid)
            return None # Status: Nowy (referencja zapisana)
        
        reference_grid = np.load(ref_path)
        diff = np.abs(current_grid - reference_grid)
        return diff > self.threshold

    def get_dominant_color(self, cell_img_bgr):
        """Oblicza średni kolor BGR dla małego kawałka."""
        # Średnia z każdego kanału
        avg_color = np.mean(cell_img_bgr, axis=(0, 1)) 
        return avg_color.astype(int) # Zwraca np. [255, 120, 10]
    
    def identify_color_name(self, bgr_color):
        """Zamienia kolor BGR na przybliżoną nazwę słowną."""
        # Konwersja pojedynczego koloru (1x1 piksel) do HSV
        pixel = np.uint8([[bgr_color]])
        hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]
        
        h, s, v = hsv

        # Logika rozpoznawania na podstawie kąta Hue (0-180 w OpenCV)
        if s < 40: return "Szarawy/Biały" # Małe nasycenie = brak wyraźnego koloru
        if v < 40: return "Czarny/Ciemny"  # Mała jasność
        
        if h < 10 or h > 160: return "Czerwony"
        elif h < 25:          return "Pomarańczowy"
        elif h < 35:          return "Żółty"
        elif h < 75:          return "Zielony"
        elif h < 130:         return "Niebieski"
        elif h < 160:         return "Fioletowy"
        
        return "Nieokreślony"
    

class Visualizer:
    """Odpowiada za rysowanie wyników na obrazie."""
    @staticmethod
    def draw_panel_info(overlay, panel, panel_id, changed_mask):
        # Kolorystyka
        if changed_mask is None:
            color = (255, 0, 0) # Niebieski - Nowy/Ref
            status_txt = "REF SAVED"
        elif np.any(changed_mask):
            color = (0, 0, 255) # Czerwony - Błąd
            status_txt = "ERR"
            Visualizer._draw_error_grid(overlay, panel, changed_mask)
        else:
            color = (0, 255, 0) # Zielony - OK
            status_txt = "OK"

        # Rysowanie obrysu i tekstów
        cv2.drawContours(overlay, [panel.box.astype(int)], 0, color, 2)
        y_pos = int(panel.box[0][1])
        cv2.putText(overlay, f"ID:{panel_id} {status_txt}", (int(panel.box[0][0]), y_pos-10), 0, 0.6, color, 2)
        cv2.putText(overlay, f"{panel.angle:.1f} deg", (int(panel.box[0][0]), y_pos-30), 0, 0.5, (0, 255, 255), 1)

    @staticmethod
    def _draw_error_grid(overlay, panel, changed_mask):
        gh, gw = changed_mask.shape
        step_h, step_w = panel.target_size[1] // gh, panel.target_size[0] // gw
        
        for i, j in np.argwhere(changed_mask):
            # Współrzędne komórki w przestrzeni "wyprostowanej"
            cell_pts = np.array([[[j*step_w, i*step_h]], [[(j+1)*step_w, i*step_h]], 
                                 [[(j+1)*step_w, (i+1)*step_h]], [[j*step_w, (i+1)*step_h]]], dtype="float32")
            
            # Mapowanie z powrotem na perspektywę klatki
            mapped_pts = cv2.perspectiveTransform(cell_pts, panel.M_inv).astype(int)
            
            # Rysowanie półprzezroczystego błędu
            temp = overlay.copy()
            cv2.fillPoly(temp, [mapped_pts], (0, 0, 255))
            cv2.addWeighted(temp, 0.4, overlay, 0.6, 0, overlay)
    
    @staticmethod
    def show(window_name, image, max_width=1280, max_height=720):
        """Skaluje obraz do podanych wymiarów i wyświetla okno."""
        h, w = image.shape[:2]
        
        # Obliczanie skali, aby zmieścić obraz w zadanym oknie
        scale = min(max_width / w, max_height / h)
        
        # Skalujemy tylko, jeśli obraz jest faktycznie większy niż limit
        if scale < 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            resized = image

        cv2.imshow(window_name, resized)
    
    @staticmethod
    def draw_cell_border(overlay, panel, cell_coords, color_bgr, label):
        """Rysuje obramowanie konkretnej komórki na oryginalnym obrazie."""
        x, y, w, h = cell_coords
        # Definiujemy punkty narożne komórki w przestrzeni "wyprostowanej"
        cell_pts = np.array([
            [[x, y]], [[x + w, y]], 
            [[x + w, y + h]], [[x, y + h]]
        ], dtype="float32")
        
        # Mapujemy punkty na oryginalną klatkę
        mapped_pts = cv2.perspectiveTransform(cell_pts, panel.M_inv).astype(int)
        
        # Rysujemy obramowanie komórki
        cv2.polylines(overlay, [mapped_pts], isClosed=True, color=color_bgr, thickness=1)
        
        # Opcjonalnie: mały napis z kolorem przy komórce (tylko jeśli chcesz bardzo gęsty opis)
        # cv2.putText(overlay, label, (mapped_pts[0][0][0], mapped_pts[0][0][1]), 
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.3, color_bgr, 1)

