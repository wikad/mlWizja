import os
import cv2
import numpy as np
from src.Panel import Panel

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