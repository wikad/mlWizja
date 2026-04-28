# System Analizy Paneli Fotowoltaicznych

Projekt detekcji i analizy paneli PV na obrazie z użyciem modelu YOLO i analizy siatki w komórkach.

## Struktura projektu

- `Main.py` - punkt wejścia programu.
- `src/Panel.py` - klasa `Panel`: detekcja prostokąta, perspektywa, podział na komórki.
- `src/PanelDetector.py` - klasa `PanelDetector`: ładowanie modelu YOLO (`ultralytics.YOLO`), maski, kontury, filtrowanie obszarów.
- `src/PanelAnalyzer.py` - klasa `PanelAnalyzer`: siatka 10x10 (domyślnie), referencje `refs/panel_{id}.npy`, wykrycie zmiany lokalnej jasności, analiza koloru komórki.
- `src/Visualizer.py` - klasy `Visualizer`: rysowanie obwiedni panelu, kolorowych komórek, podświetlanie anomalii, okienko podglądu.
- `src/Color.py` - pusty placeholder (możesz użyć dla własnych wzorców kolorów).
- `src/kolory.py` - eksperymentalna detekcja kolorów HSV w siatce komórek i wypisywanie detekcji.
- `src/m2.py` - alternatywna/duplikatowa wersja klas (Panel, PanelDetector, PanelAnalyzer, Visualizer) - nie używana w `Main.py`.

## Wymagania

Zainstaluj wymagane biblioteki:

```bash
pip install -r req.txt
```

Potrzebny jest model w `Model/best.pt` (YOLO/Ultralytics). Dostępne są foldery z danymi `Dane/` i obrazy testowe w `Obrazy/`.

## Uruchomienie

W `Main.py`:

1. `detector = pd.PanelDetector("Model/best.pt", conf=0.2)`
2. `analyzer = pa.PanelAnalyzer()`
3. `img = cv2.imread(r"Obrazy\Test1.jpg")`
4. `detected_panels = detector.detect_panels(img)`

dla każdego panelu:
- `panel.extract_warped_bgr(img)` - wyprostowany kolor
- `panel.extract_warped(img)` - wyprostowany odcienie szarości
- `analyzer.get_grid_stats(warped_gray)` i `check_changes()`
- `analyzer.get_dominant_color()` i `identify_color_name()` dla komórki
- `Visualizer.draw_cell_border()` + `draw_panel_info()`

Po zaaplikowaniu wszystkich paneli:
- `Visualizer.show("System Analizy Paneli", output)`

## Działanie logiczne

- `PanelDetector.detect_panels`: wykorzystuje `YOLO` do maski, spaja maski i znajduje kontury.
- `Panel`: oblicza `rect`, ładuje transformację `M`, pozwala wyciąć wyprostowany panel.
- `PanelAnalyzer`: pierwsze uruchomienie zapisuje referencję, kolejne porównuje i zwraca boolean 2D dla przekroczeń `threshold`.
- `Visualizer`: rysuje obrysy, statusy i kolorowe/niebieskie/czerwone alarmy.

## Notatki

- Folder `refs/` używany do metryk referencyjnych w `PanelAnalyzer`.
- Limit siatki można modyfikować w `PanelAnalyzer(grid_size=(liczba, liczba))` i `Panel.get_cells(rows, cols)`.
- W `Main.py` występuje podwójne wywołanie pętli rysowania paneli, możesz to zwinąć do jednej pętli (oszczędność wydajności).

## Rozszerzenia

- Dodaj obsługę wideo zamiast pliku obrazu.
- Doprecyzuj `src/Color.py` lub usuń jeśli nieużywane.
- Zdefiniuj strukturę ROI i kalibrację do własnych paneli.
- Zastąp `print(...)` logowaniem do pliku.

---

### Komentarz modułu `src/kolory.py`

Funkcja `detect_colors_on_panel` spekuluje, że w komórce jest jeden kolor z listy (pomarańcz, żółty, zielony, niebieski, fiolet, czerwony).
Wypisuje logi czasowe i pozycję wykrytej zmiany. Uruchomienie odbywa się przy odczycie obrazu z `sciezka_do_pliku`.

### Komentarz modułu `src/m2.py`

Powielona funkcjonalność `Panel/PanelDetector/PanelAnalyzer/Visualizer` z pliku głównego. Przydatne jako backup/refaktor, możesz połączyć ją z głównym kodem lub usunąć duplikację.
