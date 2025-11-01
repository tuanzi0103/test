import os
import streamlit as st
import pandas as pd
from services.analytics import load_all
from services.db import get_db
from services.ingestion import ingest_excel, ingest_csv, init_db_from_drive_once
from charts.high_level import show_high_level
from charts.sales_report import show_sales_report
from charts.inventory import show_inventory
from charts.product_mix_only import show_product_mix_only
from charts.customer_segmentation import show_customer_segmentation
from init_db import init_db
import subprocess
import sys
from services.ingestion import ingest_from_drive_all
import platform
import numpy as np

st.set_page_config(
    page_title="Vie Manly Analytics",
    layout="wide",
    initial_sidebar_state="auto"
)

# 关闭文件监控，避免 Streamlit Cloud 报 inotify 错误
os.environ["WATCHDOG_DISABLE_FILE_WATCH"] = "true"

# ✅ 确保 SQLite 文件和表结构存在
init_db()  # 必须先初始化数据库表结构

# ✅ 如果是空库 → 从 Google Drive 导入（现在表已经存在）
init_db_from_drive_once()

st.set_page_config(page_title="Manly Farm Dashboard", layout="wide")
st.markdown("<h1 style='font-size:26px; font-weight:700;'>📊 Vie Manly Dashboard</h1>", unsafe_allow_html=True)

# ✅ 缓存数据库加载
@st.cache_data(show_spinner="loading...")
def load_db_cached(days=365):
    db = get_db()
    return load_all(db=db)


# === 数据加载 ===
tx, mem, inv = load_db_cached()

# === Sidebar ===
st.sidebar.header("⚙️ Settings")

# 文件上传 - 添加上传状态跟踪
if "uploaded_file_names" not in st.session_state:
    st.session_state.uploaded_file_names = set()

uploaded_files = st.sidebar.file_uploader(
    "Upload files",
    type=["csv", "xlsx"],
    accept_multiple_files=True
)

# ✅ 修复上传逻辑：避免重复上传
if uploaded_files:
    db = get_db()
    new_files_uploaded = False

    for f in uploaded_files:
        # 检查文件是否已经上传过
        if f.name not in st.session_state.uploaded_file_names:
            try:
                if f.name.lower().endswith(".xlsx"):
                    ingest_excel(f)
                    new_files_uploaded = True
                    st.session_state.uploaded_file_names.add(f.name)
                    st.sidebar.info(f"📥 Processing: {f.name}")
                elif f.name.lower().endswith(".csv"):
                    ingest_csv(f)
                    new_files_uploaded = True
                    st.session_state.uploaded_file_names.add(f.name)
                    st.sidebar.info(f"📥 Processing: {f.name}")
            except Exception as e:
                st.sidebar.error(f"❌ Error processing {f.name}: {e}")
        else:
            st.sidebar.warning(f"⚠️ {f.name} already uploaded")

    if new_files_uploaded:
        st.sidebar.success("✅ Files ingested & uploaded to Google Drive.")
        # 清理缓存 → 重新加载数据库
        load_db_cached.clear()
        tx, mem, inv = load_db_cached()
        # 设置刷新标志防止死循环
        if "reloaded" not in st.session_state:
            st.session_state["reloaded"] = True
            st.rerun()
        else:
            del st.session_state["reloaded"]

# === 清空数据库 ===
if st.sidebar.button("🗑️ Clear Database"):
    conn = get_db()
    cur = conn.cursor()
    for table in ["transactions", "inventory", "members"]:
        try:
            cur.execute(f"DELETE FROM {table}")
        except Exception:
            pass
    conn.commit()
    # 清空上传记录
    st.session_state.uploaded_file_names = set()
    st.sidebar.success("✅ Database cleared!")
    load_db_cached.clear()
    st.rerun()

# === 重启应用按钮 ===
if st.sidebar.button("🔄 Restart & Reload App"):
    try:
        # 1. 清除 Streamlit 缓存
        st.cache_data.clear()
        st.cache_resource.clear()

        # 2. 清空上传状态
        if "uploaded_file_names" in st.session_state:
            del st.session_state.uploaded_file_names

        # 3. 重新从 Google Drive 导入所有数据（包括新上传的）
        st.sidebar.info("🔄 Reloading data from Google Drive...")
        ingest_from_drive_all()

        # 4. 重新加载数据
        load_db_cached.clear()
        tx, mem, inv = load_db_cached()

        st.sidebar.success("✅ App restarted with latest data!")
        st.rerun()

    except Exception as e:
        st.sidebar.error(f"❌ Restart failed: {e}")

# === 单位选择 ===
st.sidebar.subheader("📏 Units")

if inv is not None and not inv.empty and "Unit" in inv.columns:
    units_available = sorted(inv["Unit"].dropna().unique().tolist())
else:
    units_available = ["Gram 1.000", "Kilogram 1.000", "Milligram 1.000"]

conn = get_db()
try:
    rows = conn.execute("SELECT name FROM units").fetchall()
    db_units = [r[0] for r in rows]  # 修复这里的索引错误
except Exception:
    db_units = []

all_units = sorted(list(set(units_available + db_units)))
unit = st.sidebar.selectbox("Choose unit", all_units)

new_unit = st.sidebar.text_input("Add new unit")
if st.sidebar.button("➕ Add Unit"):
    if new_unit and new_unit not in all_units:
        conn.execute("CREATE TABLE IF NOT EXISTS units (name TEXT UNIQUE)")
        conn.execute("INSERT OR IGNORE INTO units (name) VALUES (?)", (new_unit,))
        conn.commit()
        st.sidebar.success(f"✅ Added new unit: {new_unit}")
        st.rerun()

# === Section 选择 ===
section = st.sidebar.radio("📂 Sections", [
    "High Level report",
    "Sales report by category",
    "Inventory",
    "product mix",
    "Customers insights"
])

# === 主体展示 ===
if section == "High Level report":
    show_high_level(tx, mem, inv)
elif section == "Sales report by category":
    show_sales_report(tx, inv)
elif section == "Inventory":
    show_inventory(tx, inv)
elif section == "product mix":
    show_product_mix_only(tx)
elif section == "Customers insights":
    show_customer_segmentation(tx, mem)
