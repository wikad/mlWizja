import cv2
import numpy as np
 
class Visualizer:
    """Odpowiada za rysowanie wyników na obrazie."""
    @staticmethod
    def draw_panel_info(overlay, panel, panel_id, changed_mask):
        # Kolorystyka
        if changed_mask is None:
            color = (255, 190, 60) # Jasnoniebieski - nowy panel, zapisano referencję
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

        # Podpis przy górnym narożniku panelu, przycięty do granic obrazu
        h, w = overlay.shape[:2]
        x = int(np.clip(panel.box[:, 0].min(), 5, w - 200))
        y = int(np.clip(panel.box[:, 1].min(), 45, h - 5))
        Visualizer._label(overlay, f"ID:{panel_id} {status_txt}", (x, y - 10), color, 0.6, 2)
        Visualizer._label(overlay, f"{panel.angle:.1f} deg", (x, y - 32), (0, 255, 255), 0.5, 1)

    @staticmethod
    def _label(img, text, org, color, scale, thickness):
        """Tekst z ciemnym tłem, czytelny na każdym podłożu."""
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        x, y = org
        cv2.rectangle(img, (x - 3, y - th - 4), (x + tw + 3, y + base), (20, 20, 20), -1)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

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
