# SEFFER — Автоматическое переименование фото (Double YOLO)

Интеллектуальная система для обработки фотографий: детекция объектов и распознавание текста (OCR).

## 🚀 Быстрый запуск через Docker

Вам не нужно устанавливать Python или зависимости. Просто запустите готовую сборку из облака:

`bash`
`docker run -p 8501:8501 -v "C:\ВАШ_ПУТЬ_К_ФОТО:/app/output" ghcr.io/olgam-ds/seffer_auto_naming:latest`

Важно: Замените C:\ВАШ_ПУТЬ_К_ФОТО на реальный путь к папке на вашем компьютере, куда вы хотите сохранять результаты обработки.

## 🛠 Технологии
YOLOv8 (Object Detection)
OCR (Text Recognition)
Streamlit (Web Interface)
Docker (Containerization)
## 📁 Структура проекта
app.py — веб-интерфейс на Streamlit.
best_obb.pt / best_ocr.pt — обученные веса моделей YOLO.
pipeline.py — основная логика обработки изображений.
