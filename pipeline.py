import os
import cv2
import torch
import numpy as np
import pandas as pd
from ultralytics import YOLO
from PIL import Image, ExifTags
# ==========================================================
# 📌 НАСТРОЙКИ (Относительные пути для переносимости)
# ==========================================================
PATH_OBB_MODEL = "best_obb.pt"   # Файл должен быть в папке со скриптом
PATH_OCR_MODEL = "best_ocr.pt"   # Файл должен быть в папке со скриптом

DEBUG_MODE = True
DEBUG_CROP_DIR = "debug_crops" # Папка для кропов
DEBUG_OCR_DIR = "debug_ocr" # Папка для ocr

# ==========================================================
# 🛠 ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ГЕОМЕТРИИ
# ==========================================================
def order_points(pts):
    """Сортирует 4 точки OBB: [TL, TR, BR, BL]"""
    pts = np.array(pts, dtype="float32").reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # top-right
    rect[3] = pts[np.argmax(diff)] # bottom-left
    return rect

def get_perspective_transform(image, pts):
    """Вырезает и выравнивает табличку без жестких фильтров."""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if width == 0 or height == 0: return None
    dst = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (width, height))
    if height > width:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    return warped

# ==========================================================
# 🚀 ГЛАВНЫЙ КЛАСС ПАЙПЛАЙНА
# ==========================================================
class FinalPipeline:
    def __init__(self):
        print("🚀 Инициализация моделей...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.obb_model = YOLO(PATH_OBB_MODEL).to(device)
        self.ocr_model = YOLO(PATH_OCR_MODEL).to(device)
        
        if DEBUG_MODE:
            os.makedirs(DEBUG_CROP_DIR, exist_ok=True)
            os.makedirs(DEBUG_OCR_DIR, exist_ok=True)

    def _fix_orientation(self, path):
        """Этап 0: Поворот фото по EXIF"""
        try:
            img = Image.open(path)
            exif = img._getexif()
            if exif:
                orientation = next((k for k, v in ExifTags.TAGS.items() if v == 'Orientation'), None)
                if orientation and orientation in exif:
                    v = exif[orientation]
                    if v == 3: img = img.rotate(180, expand=True)
                    elif v == 6: img = img.rotate(270, expand=True)
                    elif v == 8: img = img.rotate(90, expand=True)
            return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        except:
            return cv2.imread(path)

    def _process_ocr(self, crop, filename):
        """Этап 3: OCR с финальной геометрической фильтрацией и визуализацией"""
        # Порог 0.15 для поиска, iou=0.1 для схлопывания наложений
        results = self.ocr_model.predict(crop, conf=0.15, iou=0.1, verbose=False)
        res = results[0] 
        if res.boxes is None or len(res.boxes) == 0:
            return "!" + filename, 0.0
            
        h_img, w_img = crop.shape[:2]
        raw_chars = []
        for box in res.boxes:
            b = box.xyxy[0].cpu().numpy()
            w_box, h_box = b[2] - b[0], b[3] - b[1]
            conf = box.conf[0].item()
            x_center = box.xywh[0][0].item()
            
            # --- ФИЛЬТРЫ ГЕОМЕТРИИ (Убираем мусор) ---
            if w_box < h_box * 0.20 and conf < 0.6: continue
            if (x_center < w_img * 0.10 or x_center > w_img * 0.90) and conf < 0.75: continue
            if h_box < h_img * 0.25: continue

            raw_chars.append({'x': x_center, 'val': str(int(box.cls[0].item())), 'conf': conf, 'bbox': b})

        if not raw_chars: return "!" + filename, 0.0

        # Склеиваем дубликаты (если рамки слишком близко)
        raw_chars.sort(key=lambda x: x['conf'], reverse=True)
        final_chars = []
        for char in raw_chars:
            if not any(abs(char['x'] - a['x']) < (w_img * 0.04) for a in final_chars):
                final_chars.append(char)

        # Сортировка слева направо для сборки числа
        final_chars.sort(key=lambda x: x['x'])
        raw_text = "".join([c['val'] for c in final_chars])
        avg_conf = np.mean([c['conf'] for c in final_chars])
        
        # --- ВИЗУАЛИЗАЦИЯ ОТЛАДКИ ---
        if DEBUG_MODE:
            debug_img = crop.copy()
            for c in final_chars:
                b_int = c['bbox'].astype(int)
                # Рисуем рамку
                cv2.rectangle(debug_img, (b_int[0], b_int[1]), (b_int[2], b_int[3]), (0, 255, 0), 2)
                
                # Компактный текст: "Цифра (conf)"
                display_text = f"{c['val']} ({c['conf']:.2f})"
                cv2.putText(debug_img, display_text, (b_int[0], b_int[1] - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                
            cv2.imwrite(os.path.join(DEBUG_OCR_DIR, f"ocr_{filename}"), debug_img)
        
        if 0 < len(raw_text) <= 4:
            return raw_text.zfill(4), round(float(avg_conf), 2)
        return "!" + filename, round(float(avg_conf), 2)

    def run(self, prefix, image_folder):
        """Главный метод для вызова из Streamlit"""
        results = []
        counter = 1
        files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"🎬 Партия {prefix}: Обработка {len(files)} файлов...")

        for fs in files:
            img_path = os.path.join(image_folder, fs)
            img = self._fix_orientation(img_path)
            
            # Детекция OBB (Порог 0.25 для охвата сложных случаев)
            results_obb = self.obb_model.predict(img, conf=0.25, verbose=False)
            res_obb = results_obb[0]
            
            if res_obb.obb is None or len(res_obb.obb.conf) == 0:
                final_val, final_conf = "!" + fs, 0.0
            else:
                best_idx = res_obb.obb.conf.cpu().numpy().argmax()
                pts = res_obb.obb.xyxyxyxy[best_idx].cpu().numpy()
                crop = get_perspective_transform(img, pts)
                
                if crop is None:
                    final_val, final_conf = "!" + fs, 0.0
                else:
                    if DEBUG_MODE:
                        cv2.imwrite(os.path.join(DEBUG_CROP_DIR, f"crop_{fs}"), crop)
                    final_val, final_conf = self._process_ocr(crop, fs)
            
            # Твои новые названия столбцов
            results.append({
                "id": f"{prefix}_{counter}",
                "old_filename": fs,
                "result": final_val,
                "confidence": final_conf
            })
            counter += 1
        
        df = pd.DataFrame(results)
        df.to_csv(f"Report_{prefix}.csv", index=False, encoding='utf-8-sig')
        return df

if __name__ == "__main__":
    # Тестовый запуск
    p = FinalPipeline()
    p.run("TEST", "uploaded_photos")
