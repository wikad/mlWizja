import os
import cv2
import numpy as np
from ultralytics import YOLO
from src.Panel import Panel

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