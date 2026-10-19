"""
Detekcja paneli fotowoltaicznych i wykrywanie zmian na ich powierzchni.

Użycie:
    python Main.py <obraz> [--save wynik.jpg] [--no-show] [--conf 0.2] [--ref-dir refs]

Pierwsze uruchomienie dla danej instalacji zapisuje obraz referencyjny każdego panelu
(status REF SAVED). Kolejne porównują stan paneli z referencją i zaznaczają komórki,
w których jasność zmieniła się powyżej progu (zabrudzenie, uszkodzenie, cień).
"""
import argparse
import sys

import cv2

import src.PanelAnalyzer as pa
import src.PanelDetector as pd
import src.Visualizer as vs

GRID = 10  # panel dzielony na siatkę GRID x GRID komórek


def parse_args():
    parser = argparse.ArgumentParser(description="Analiza paneli PV na obrazie")
    parser.add_argument("image", help="ścieżka do obrazu")
    parser.add_argument("--model", default="Model/best.pt", help="model segmentacji YOLO")
    parser.add_argument("--conf", type=float, default=0.2, help="próg pewności detekcji")
    parser.add_argument("--ref-dir", default="refs", help="folder z referencjami paneli")
    parser.add_argument("--threshold", type=float, default=25, help="próg zmiany jasności komórki")
    parser.add_argument("--save", help="zapisz wynik do pliku")
    parser.add_argument("--no-show", action="store_true", help="nie otwieraj okna podglądu")
    return parser.parse_args()


def main():
    args = parse_args()
    detector = pd.PanelDetector(args.model, conf=args.conf)
    analyzer = pa.PanelAnalyzer(ref_dir=args.ref_dir, grid_size=(GRID, GRID), threshold=args.threshold)

    img = cv2.imread(args.image)
    if img is None:
        sys.exit(f"Nie można wczytać obrazu: {args.image}")

    output = img.copy()
    panels = detector.detect_panels(img)
    print(f"Wykryto paneli: {len(panels)}")

    for i, panel in enumerate(panels):
        warped_bgr = panel.extract_warped_bgr(img)
        warped_gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)

        # Porównanie siatki jasności z referencją panelu
        changes = analyzer.check_changes(i, analyzer.get_grid_stats(warped_gray))

        # Ramka każdej komórki w jej średnim kolorze, czerwona tam, gdzie wykryto zmianę
        for cell_img, r, c, coords in panel.get_cells(warped_bgr, rows=GRID, cols=GRID):
            if changes is not None and changes[r, c]:
                color = (0, 0, 255)
            else:
                color = tuple(map(int, analyzer.get_dominant_color(cell_img)))
            vs.Visualizer.draw_cell_border(output, panel, coords, color, None)

        vs.Visualizer.draw_panel_info(output, panel, i, changes)

        status = "referencja zapisana" if changes is None else f"zmienione komórki: {int(changes.sum())}/{GRID * GRID}"
        print(f"  Panel {i}: {panel}, {status}")

    if args.save:
        cv2.imwrite(args.save, output)
        print(f"Zapisano: {args.save}")

    if not args.no_show:
        vs.Visualizer.show("System Analizy Paneli", output)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
