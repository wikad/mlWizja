import cv2

# Importujemy z nowego folderu 'src'
import src.Panel as p
import src.PanelDetector as pd
import src.PanelAnalyzer as pa
import src.Visualizer as vs
import src.Color as co

# ====== (MAIN LOOP) ======
if __name__ == "__main__":
    detector = pd.PanelDetector("Model/best.pt", conf = 0.2)
    analyzer = pa.PanelAnalyzer()
    
    img = cv2.imread(r"Obrazy\Test1.jpg")
    if img is None: exit()
    
    output = img.copy()
    detected_panels = detector.detect_panels(img)
    
    for i, panel in enumerate(detected_panels):
        warped_bgr = panel.extract_warped_bgr(img)
        warped_gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
        
        # Pobieramy zmiany (siatka 10x10)
        current_grid = analyzer.get_grid_stats(warped_gray)
        changes = analyzer.check_changes(i, current_grid)
        
        # Analiza kolorów (używamy tej samej siatki 10x10 co w analyzerze!)
        for cell_img, r, c, coords in panel.get_cells(warped_bgr, rows=10, cols=10):
            avg_bgr = analyzer.get_dominant_color(cell_img)
            color_name = analyzer.identify_color_name(avg_bgr)
            bgr_tuple = tuple(map(int, avg_bgr))
            
            # Jeśli w tym miejscu jest błąd (zmiana jasności > THRESHOLD)
            if changes is not None and changes[r, c]:
                # Rysujemy grubszą, czerwoną ramkę dla anomalii
                vs.Visualizer.draw_cell_border(output, panel, coords, (0, 0, 255), color_name)
                print(f"Panel {i} Cell ({r},{c}): ZMIANA! Obecny kolor: {color_name}")
            else:
                # Rysujemy zwykłą cienką ramkę w kolorze komórki
                vs.Visualizer.draw_cell_border(output, panel, coords, bgr_tuple, color_name)

        vs.Visualizer.draw_panel_info(output, panel, i, changes)


    for i, panel in enumerate(detected_panels):
        warped_gray = panel.extract_warped(img)
        current_grid = analyzer.get_grid_stats(warped_gray)
        changes = analyzer.check_changes(i, current_grid)
        
        vs.Visualizer.draw_panel_info(output, panel, i, changes)

    vs.Visualizer.show("System Analizy Paneli", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()