import cv2
import numpy as np
 
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
