# Используем Python 3.9
FROM python:3.9-slim

# Устанавливаем системные библиотеки для OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Рабочая папка
WORKDIR /app

# Создаем все необходимые папки заранее
RUN mkdir -p /app/output /app/uploaded_photos /app/renamed_photos /app/debug_crops /app/debug_ocr

# Копируем зависимости и ставим их без лишнего мусора
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем проект
COPY . .

# Порт
EXPOSE 8501

# Запуск
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]