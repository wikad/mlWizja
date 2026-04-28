import cv2
import numpy as np
import datetime

def detect_colors_on_panel(panel_img, panel_name, angle, liczba_kolumn=10, liczba_wierszy=10):
    wysokosc, szerokosc, _ = panel_img.shape
    
    szer_komorki = szerokosc // liczba_kolumn
    wys_komorki = wysokosc // liczba_wierszy
    powierzchnia_komorki = szer_komorki * wys_komorki
    
    PROG_WYKRYCIA = 0.15 
    
    hsv_img = cv2.cvtColor(panel_img, cv2.COLOR_BGR2HSV)
    
    kolory_hsv = {
        "POMARAŃCZOWY": (np.array([10, 100, 100]), np.array([20, 255, 255])),
        "ŻÓŁTY":        (np.array([20, 100, 100]), np.array([35, 255, 255])),
        "ZIELONY":      (np.array([35, 100, 100]), np.array([85, 255, 255])),
        "NIEBIESKI":    (np.array([90, 100, 100]), np.array([120, 255, 255])),
        
        "FIOLETOWY":    (np.array([116, 0, 133]),  np.array([126, 255, 184]))
    }
    
    
    czerwony_dol_1 = np.array([0, 100, 100])
    czerwony_gora_1 = np.array([10, 255, 255])
    czerwony_dol_2 = np.array([170, 100, 100])
    czerwony_gora_2 = np.array([179, 255, 255])

    
    for y_droniada in range(liczba_wierszy):
        for x_droniada in range(liczba_kolumn):
            
            y_opencv = (liczba_wierszy - 1) - y_droniada
            x_opencv = x_droniada
            
            y_start = y_opencv * wys_komorki
            y_end = y_start + wys_komorki
            x_start = x_opencv * szer_komorki
            x_end = x_start + szer_komorki
            
            komorka_hsv = hsv_img[y_start:y_end, x_start:x_end]
            wykryty_kolor = None
            
            for nazwa_koloru, (dol, gora) in kolory_hsv.items():
                maska = cv2.inRange(komorka_hsv, dol, gora)
                liczba_pikseli = cv2.countNonZero(maska)
                
                if liczba_pikseli > (powierzchnia_komorki * PROG_WYKRYCIA):
                    wykryty_kolor = nazwa_koloru
                    break 
            
            if not wykryty_kolor:
                maska1 = cv2.inRange(komorka_hsv, czerwony_dol_1, czerwony_gora_1)
                maska2 = cv2.inRange(komorka_hsv, czerwony_dol_2, czerwony_gora_2)
                maska_czerwona = cv2.bitwise_or(maska1, maska2)
                
                if cv2.countNonZero(maska_czerwona) > (powierzchnia_komorki * PROG_WYKRYCIA):
                    wykryty_kolor = "CZERWONY"
            
            if wykryty_kolor:
                czas = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"[{czas}] WYKRYTO ZMIANĘ -> Panel: {panel_name} ({angle}°) | Pozycja: Wiersz {y_droniada}, Kolumna {x_droniada} | Kolor: {wykryty_kolor}")


sciezka_do_pliku = 'TUTAJ_WPISZ_NAZWE_PLIKU.png' 
    
img = cv2.imread(sciezka_do_pliku)

if img is None:
    print(f"BŁĄD: Nie można wczytać pliku! Upewnij się, że plik '{sciezka_do_pliku}' znajduje się w folderze z kodem.")
else:

    detect_colors_on_panel(
        panel_img=img, 
        panel_name="A", 
        angle="0", 
        liczba_kolumn=10, 
        liczba_wierszy=10 
    )