import openpyxl
import streamlit as st
from pipeline import FinalPipeline
import pandas as pd
import os
import shutil
from datetime import datetime
from PIL import Image

# --- ПУТИ (внутренние для контейнера) ---
UPLOAD_DIR = "uploaded_photos"
RESULT_DIR = "renamed_photos"
DEBUG_CROP_DIR = "debug_crops" 
DEBUG_OCR_DIR = "debug_ocr"
INTERNAL_EXPORT_PATH = "/app/output"

for d in [UPLOAD_DIR, RESULT_DIR, INTERNAL_EXPORT_PATH]:
    os.makedirs(d, exist_ok=True)

# --- ФУНКЦИИ ---
def get_clean_name(text_val):
    if isinstance(text_val, tuple):
        text_val = text_val[0]
    return os.path.splitext(str(text_val))[0]

def get_unique_path(directory, name, ext):
    ext = f".{ext.strip('.')}"
    name_only = get_clean_name(name)
    full_name = f"{name_only}{ext}"
    path = os.path.join(directory, full_name)
    counter = 1
    while os.path.exists(path):
        full_name = f"{name_only}_{counter}{ext}"
        path = os.path.join(directory, full_name)
        counter += 1
    return path, full_name

def log_result_to_batch(old_name, new_name, mode, row_id, log_path):
    new_data = pd.DataFrame([{
        "Дата": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ID_Пайплайна": row_id,
        "Исходное имя": old_name,
        "Финальное имя": new_name,
        "Тип правки": mode
    }])
    if os.path.exists(log_path):
        try:
            df = pd.read_excel(log_path)
            df = pd.concat([df, new_data], ignore_index=True)
            df.to_excel(log_path, index=False)
        except:
            new_data.to_excel(log_path, index=False)
    else:
        new_data.to_excel(log_path, index=False)

def clean_all_work_folders():
    folders = [UPLOAD_DIR, RESULT_DIR, DEBUG_CROP_DIR, DEBUG_OCR_DIR]
    for folder in folders:
        if not os.path.exists(folder): continue
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path): os.unlink(file_path)
                elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except: pass
    for f in os.listdir("."):
        if f.startswith("Report_"):
            try: os.unlink(f)
            except: pass

@st.cache_resource
def load_pipe():
    return FinalPipeline()

pipeline = load_pipe()

# --- ИНТЕРФЕЙС ---
st.set_page_config(layout="wide", page_title="СЭФФЕР")

if 'stage' not in st.session_state: st.session_state.stage = 'upload'
if 'prefix' not in st.session_state: st.session_state.prefix = "Sector_Default"
if 'results_dict' not in st.session_state: st.session_state.results_dict = {}
if 'manual_mode' not in st.session_state: st.session_state.manual_mode = False
if 'export_done' not in st.session_state: st.session_state.export_done = False
if 'last_processed_img' not in st.session_state: st.session_state.last_processed_img = None

files_in_upload = sorted(os.listdir(UPLOAD_DIR)) if os.path.exists(UPLOAD_DIR) else []

# --- ВОССТАНОВЛЕНИЕ ---
if st.session_state.stage == 'upload' and files_in_upload:
    st.warning("📂 **Обнаружена незавершенная работа!**")
    report_file = next((f for f in os.listdir(".") if f.startswith("Report_") and f.endswith(".csv")), None)
    found_prefix = report_file.replace("Report_", "").replace(".csv", "") if report_file else "Неизвестно"
    st.write(f"Партия: **{found_prefix}** | Осталось: **{len(files_in_upload)}** фото.")
    col_rec1, col_rec2 = st.columns(2)
    with col_rec1:
        if st.button("🚀 Продолжить работу", use_container_width=True):
            st.session_state.prefix = found_prefix
            if report_file:
                df_temp = pd.read_csv(report_file)
                st.session_state.results_dict = {row['old_filename']: {'text': row['result'], 'id': row['id']} for _, row in df_temp.iterrows()}
            st.session_state.stage = 'validation'
            st.rerun()
    with col_rec2:
        if st.button("🗑️ Сбросить и начать заново", use_container_width=True):
            clean_all_work_folders()
            st.session_state.clear()
            st.rerun()
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.title("СЭФФЕР")
    if st.session_state.stage == 'validation' and len(files_in_upload) > 0:
        st.subheader("Контекст")
        if st.session_state.last_processed_img:
            st.caption("⬅️ Предыдущее фото")
            st.image(st.session_state.last_processed_img, use_container_width=True)
        if len(files_in_upload) > 1:
            st.caption("➡️ Следующее фото")
            st.image(Image.open(os.path.join(UPLOAD_DIR, files_in_upload[1])), use_container_width=True)
    st.divider()
    st.error("Внимание! При сбросе все данные будут удалены.")
    if st.button("🔄 Начать всё заново", use_container_width=True):
        clean_all_work_folders()
        st.session_state.clear()
        st.rerun()

st.title("СЭФФЕР: Переименование фото")
batch_log_path = os.path.join(RESULT_DIR, f"{st.session_state.prefix}_report.xlsx")

# --- ШАГ 1: ЗАГРУЗКА ---
if st.session_state.stage == 'upload':
    st.subheader("Шаг 1: Подготовка данных")
    uploaded_files = st.file_uploader("Выберите фото (Ctrl+A в папке)", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
    if uploaded_files:
        st.info(f"✅ К загрузке готово: {len(uploaded_files)} файлов")
    st.session_state.prefix = st.text_input("Задайте ПРЕФИКС партии:", st.session_state.prefix)
    if st.button("📥 Начать загрузку"):
        if uploaded_files:
            clean_all_work_folders() 
            for uf in uploaded_files:
                with open(os.path.join(UPLOAD_DIR, uf.name), "wb") as f:
                    f.write(uf.getbuffer())
            st.session_state.stage = 'processing'
            st.rerun()
        else:
            st.error("Файлы не выбраны!")

# --- ШАГ 2: ОБРАБОТКА ---
elif st.session_state.stage == 'processing':
    st.subheader(f"Шаг 2: Распознавание партии {st.session_state.prefix}")
    if st.button("🚀 ЗАПУСТИТЬ DOUBLE YOLO"):
        with st.spinner("Нейронка работает..."):
            df_results = pipeline.run(st.session_state.prefix, UPLOAD_DIR)
            if df_results is not None and not df_results.empty:
                st.session_state.results_dict = {row['old_filename']: {'text': row['result'], 'id': row['id']} for _, row in df_results.iterrows()}
                st.session_state.stage = 'validation'
                st.rerun()

# --- ШАГ 3: ВАЛИДАЦИЯ ---
elif st.session_state.stage == 'validation':
    if not files_in_upload:
        st.session_state.stage = 'export'
        st.rerun()
    else:
        curr_file = files_in_upload[0]
        full_path = os.path.join(UPLOAD_DIR, curr_file)
        res_data = st.session_state.results_dict.get(curr_file, {'text': '???', 'id': 'N/A'})
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Текущее фото") # 1. Подсказка сверху
            img = Image.open(full_path)
            st.image(img, caption=f"Файл: {curr_file}", use_container_width=True)
        
        with col2:
            st.subheader("Валидация")
            name_no_ext = get_clean_name(curr_file)
            clean_pred = get_clean_name(res_data.get('text', '???'))
            
            crop_path = None
            if os.path.exists(DEBUG_CROP_DIR):
                crops = [f for f in os.listdir(DEBUG_CROP_DIR) if name_no_ext in f]
                if crops: crop_path = os.path.join(DEBUG_CROP_DIR, crops[0])
            
            if crop_path: 
                st.image(Image.open(crop_path), width=250, caption="Кроп бирки")
            else:
                st.warning("Метка не найдена на фото") # 2. Подсказка если кропа нет

            st.write(f"Предсказано: **{clean_pred}**")
            
            if st.button(f"✅ Подтвердить {clean_pred}", use_container_width=True):
                st.session_state.last_processed_img = Image.open(full_path).copy()
                st.session_state.last_processed_img.thumbnail((300, 300))
                target, _ = get_unique_path(RESULT_DIR, clean_pred, ".jpg")
                shutil.move(full_path, target)
                log_result_to_batch(curr_file, os.path.basename(target), "auto", res_data['id'], batch_log_path)
                st.rerun()
            
            if st.button("✏️ Ввести вручную", use_container_width=True):
                st.session_state.manual_mode = not st.session_state.manual_mode
            
            if st.session_state.manual_mode:
                # 3. Подсказка в поле ввода (help и placeholder)
                m_val = st.text_input(
                    "Верный номер:", 
                    value=clean_pred,
                    help="Введите 4 символа без префикса, например: 0092",
                    placeholder="0092"
                )
                if st.button("💾 Сохранить"):
                    st.session_state.last_processed_img = Image.open(full_path).copy()
                    st.session_state.last_processed_img.thumbnail((300, 300))
                    target, _ = get_unique_path(RESULT_DIR, m_val, ".jpg")
                    shutil.move(full_path, target)
                    log_result_to_batch(curr_file, os.path.basename(target), "manual", res_data['id'], batch_log_path)
                    st.session_state.manual_mode = False
                    st.rerun()

# --- ШАГ 4: ВЫГРУЗКА ---
elif st.session_state.stage == 'export':
    st.subheader("🏁 Шаг 4: Выгрузка")
    add_debug = st.checkbox("Добавить тех. данные", value=True, disabled=st.session_state.export_done)
    if st.button("📦 ВЫГРУЗИТЬ ДАННЫЕ", use_container_width=True, disabled=st.session_state.export_done):
        final_folder = os.path.join(INTERNAL_EXPORT_PATH, st.session_state.prefix)
        os.makedirs(final_folder, exist_ok=True)
        for f in os.listdir(RESULT_DIR):
            shutil.move(os.path.join(RESULT_DIR, f), os.path.join(final_folder, f))
        if add_debug:
            debug_dest = os.path.join(final_folder, "debug_info")
            os.makedirs(debug_dest, exist_ok=True)
            for d in [DEBUG_CROP_DIR, DEBUG_OCR_DIR]:
                if os.path.exists(d):
                    for item in os.listdir(d): shutil.copy(os.path.join(d, item), os.path.join(debug_dest, item))
            for f in os.listdir("."):
                if f.startswith("Report_"): shutil.move(f, os.path.join(debug_dest, f))
        st.session_state.export_done = True
        st.success(f"Выгружено! Проверьте папку на диске.")
        st.balloons()
        st.rerun()
    if st.session_state.export_done:
        if st.button("🔄 НОВАЯ ПАРТИЯ"):
            clean_all_work_folders()
            st.session_state.clear()
            st.rerun()

if st.session_state.stage in ['validation', 'export'] and os.path.exists(batch_log_path):
    st.divider()
    st.dataframe(pd.read_excel(batch_log_path), use_container_width=True)
