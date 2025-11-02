# services/ingestion.py
import os
import re
import tempfile
from io import BytesIO

import pandas as pd
import streamlit as st

from services.db import get_db

# === Google Drive 相关 ===
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

FOLDER_ID = "1lZGE0DkgKyox1HbBzuhZ-oypDF478jBj"

# ✅ 全局缓存 drive 实例
_drive_instance = None


def get_drive():
    global _drive_instance
    if _drive_instance is not None:
        return _drive_instance

    gauth = GoogleAuth()

    # ✅ 设置 OAuth 为 offline，确保 refresh_token 存下来
    gauth.settings['oauth_scope'] = [
        "https://www.googleapis.com/auth/drive",
        "https://www.googleapis.com/auth/drive.file",
        "https://www.googleapis.com/auth/drive.metadata"
    ]
    gauth.settings['get_refresh_token'] = True

    token_path = "credentials.json"
    if os.path.exists(token_path):
        gauth.LoadCredentialsFile(token_path)

    if gauth.credentials is None:
        # 第一次认证：加 offline
        gauth.LocalWebserverAuth()  # 这里会弹浏览器
    elif gauth.access_token_expired:
        gauth.Refresh()
    else:
        gauth.Authorize()

    gauth.SaveCredentialsFile(token_path)
    _drive_instance = GoogleDrive(gauth)
    return _drive_instance


def upload_file_to_drive(local_path: str, remote_name: str):
    """把本地文件上传到指定的 Google Drive 文件夹。"""
    try:
        drive = get_drive()
        f = drive.CreateFile({'title': remote_name, 'parents': [{'id': FOLDER_ID}]})
        f.SetContentFile(local_path)
        f.Upload()
    except Exception as e:
        # 上传失败不影响主流程
        st.sidebar.warning(f"⚠️ Upload to Drive failed: {e}")


def download_file_from_drive(file_id, local_path):
    drive = get_drive()
    f = drive.CreateFile({'id': file_id})
    f.GetContentFile(local_path)


# --------------- 工具函数 ---------------

def _fix_header(df: pd.DataFrame) -> pd.DataFrame:
    """若第一行是 Unnamed，多数是多行表头；把第二行提为表头。"""
    if len(df.columns) and all(str(c).startswith("Unnamed") for c in df.columns):
        df.columns = df.iloc[0]
        df = df.drop(index=0).reset_index(drop=True)
    return df


def _to_float(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(r"[^0-9\.\-]", "", regex=True)
        .replace("", pd.NA)
        .astype(float)
    )


def _extract_date_from_filename(name: str):
    """从文件名中提取 YYYY-MM-DD"""
    m = re.search(r"(\d{4}-\d{2}-\d{2})", name)
    if m:
        return m.group(1)
    return None


# --------------- 预处理（不改列名） ---------------

def preprocess_transactions(df: pd.DataFrame) -> pd.DataFrame:
    df = _fix_header(df)
    if "Date" in df.columns and "Time" in df.columns:
        df["Datetime"] = pd.to_datetime(
            df["Date"].astype(str) + " " + df["Time"].astype(str),
            errors="coerce"
        )
        drop_cols = [c for c in ["Date", "Time", "Time Zone"] if c in df.columns]
        df = df.drop(columns=drop_cols)
    elif "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")

    for col in ["Net Sales", "Gross Sales", "Qty", "Discounts"]:
        if col in df.columns:
            df[col] = _to_float(df[col])

    # === 新增：Card Brand 与 PAN Suffix 处理，保证写入数据库 ===
    if "Card Brand" in df.columns:
        df["Card Brand"] = (
            df["Card Brand"]
            .astype(str)
            .str.strip()
            .str.title()  # 标准化为首字母大写
        )

    if "PAN Suffix" in df.columns:
        df["PAN Suffix"] = (
            df["PAN Suffix"]
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)  # 去掉浮点形式的".0"
            .str.strip()
        )

    return df


def preprocess_inventory(df: pd.DataFrame, filename: str = None) -> pd.DataFrame:
    df = _fix_header(df)

    # inventory表格从第二行开始是header
    if len(df) > 0 and all(str(col).startswith("Unnamed") for col in df.columns):
        df.columns = df.iloc[0]
        df = df[1:].reset_index(drop=True)
        df = _fix_header(df)  # 再次处理可能的多行表头

    required = [
        "Tax - GST (10%)", "Price", "Current Quantity Vie Market & Bar",
        "Default Unit Cost", "Categories"
    ]
    for col in required:
        if col not in df.columns:
            df[col] = None

    # 过滤掉Current Quantity Vie Market & Bar或者Default Unit Cost为空的行
    if "Current Quantity Vie Market & Bar" in df.columns and "Default Unit Cost" in df.columns:
        for col in ["Price", "Current Quantity Vie Market & Bar", "Default Unit Cost"]:
            if col not in df.columns:
                df[col] = None
            df[col] = (
                df[col].astype(str)
                .str.replace(r"[^0-9\.\-]", "", regex=True)
                .replace("", pd.NA)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if filename:
        df["source_date"] = _extract_date_from_filename(filename)
    return df


def preprocess_members(df: pd.DataFrame) -> pd.DataFrame:
    return _fix_header(df)


# --------------- 表结构对齐 & 去重 & 写入 ---------------

def _table_exists(conn, table: str) -> bool:
    try:
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
        return cur.fetchone() is not None
    except Exception:
        return False


def _existing_columns(conn, table: str) -> list:
    try:
        cur = conn.execute(f"PRAGMA table_info('{table}')")
        return [row[1] for row in cur.fetchall()]
    except Exception:
        return []


def _add_missing_columns(conn, table: str, missing_cols: list, prefer_real: set):
    cur = conn.cursor()
    for col in missing_cols:
        coltype = "REAL" if col in prefer_real else "TEXT"
        cur.execute(f'''ALTER TABLE "{table}" ADD COLUMN "{col}" {coltype}''')
    conn.commit()


def _ensure_table_schema(conn, table: str, df: pd.DataFrame, prefer_real: set):
    if not _table_exists(conn, table):
        # 如果表不存在，创建表
        df.head(0).to_sql(table, conn, if_exists="replace", index=False)
        return
    cols_now = set(_existing_columns(conn, table))
    incoming = list(df.columns)
    missing = [c for c in incoming if c not in cols_now]
    if missing:
        _add_missing_columns(conn, table, missing, prefer_real)


def _deduplicate(df: pd.DataFrame, key_col: str, conn, table: str) -> pd.DataFrame:
    """
    默认仍旧按单列 key_col 去重；
    但对 inventory 表，如果同时具备 source_date+SKU，则按 (source_date, SKU) 复合键去重。
    """
    if df is None or df.empty:
        return df

    # ✅ inventory：使用 (source_date, SKU) 去重，避免跨天误伤
    if table == "inventory" and "source_date" in df.columns and "SKU" in df.columns:
        try:
            exist = pd.read_sql('SELECT source_date, SKU FROM "inventory"', conn)
            # 统一成字符串键，避免 NaT/NaN 对比问题
            exist["source_date"] = pd.to_datetime(exist["source_date"], errors="coerce").dt.date.astype(str)
            exist["SKU"] = exist["SKU"].astype(str)
            existed_keys = set((exist["source_date"] + "||" + exist["SKU"]).unique())

            df_local = df.copy()
            df_local["source_date"] = pd.to_datetime(df_local["source_date"], errors="coerce").dt.date.astype(str)
            df_local["SKU"] = df_local["SKU"].astype(str)
            keys = df_local["source_date"] + "||" + df_local["SKU"]
            mask = ~keys.isin(existed_keys)
            return df_local[mask]
        except Exception:
            # 读库失败时，不做去重，尽量不中断导入
            return df

    # 其它表/场景：保持原单键去重逻辑
    if key_col not in df.columns:
        return df
    try:
        exist = pd.read_sql(f'''SELECT "{key_col}" FROM "{table}"''', conn)
        existed_set = set(exist[key_col].dropna().astype(str).unique())
        mask = ~df[key_col].astype(str).isin(existed_set)
        return df[mask]
    except Exception:
        return df



def _write_df(conn, df: pd.DataFrame, table: str, key_candidates: list, prefer_real: set):
    _ensure_table_schema(conn, table, df, prefer_real)

    # ✅ 修复：inventory 表按日期先删后写，避免重复累加
    if table == "inventory" and "source_date" in df.columns:
        dates = df["source_date"].dropna().unique().tolist()
        if dates:
            for d in dates:
                conn.execute(f'DELETE FROM "{table}" WHERE source_date=?', (d,))
            conn.commit()

    key_col = next((k for k in key_candidates if k in df.columns), None)
    if key_col:
        df = _deduplicate(df, key_col, conn, table)

    if not df.empty:
        df.to_sql(table, conn, if_exists="append", index=False)


# --------------- 索引 ---------------
def ensure_indexes():
    conn = get_db()
    cur = conn.cursor()

    # 确保表存在
    for table in ["transactions", "inventory", "members"]:
        if not _table_exists(conn, table):
            # 如果表不存在，创建空表
            pd.DataFrame().to_sql(table, conn, if_exists="replace", index=False)

    cur.execute('CREATE INDEX IF NOT EXISTS idx_txn_datetime ON transactions(Datetime)')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_txn_id ON transactions([Transaction ID])')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_member_square ON members([Square Customer ID])')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_member_ref ON members([Reference ID])')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_inv_sku ON inventory(SKU)')
    conn.commit()


# --------------- 从 Google Drive 导入 ---------------
def ingest_from_drive_all():
    conn = get_db()

    # 确保表存在
    ensure_indexes()

    drive = get_drive()
    files = drive.ListFile({'q': f"'{FOLDER_ID}' in parents and trashed=false"}).GetList()
    if not files:
        return

    seen = set()
    for f in files:
        name = f["title"]
        if name in seen:
            continue
        seen.add(name)

        local = os.path.join(tempfile.gettempdir(), name)
        f.GetContentFile(local)

        if name.lower().endswith(".csv"):
            df = pd.read_csv(local)
            df = _fix_header(df)
        elif name.lower().endswith(".xlsx"):
            header_row = 1 if "catalogue" in name.lower() else 0
            df = pd.read_excel(local, header=header_row)
            df = _fix_header(df)
        else:
            continue

        if "items" in name.lower():
            df = preprocess_transactions(df)
            _write_df(conn, df, "transactions",
                      key_candidates=["Transaction ID", "Item", "Price", "Modifiers Applied"],
                      prefer_real={"Net Sales", "Gross Sales", "Qty", "Discounts"})

        elif "catalogue" in name.lower():
            df = preprocess_inventory(df, filename=name)
            _write_df(conn, df, "inventory",
                      key_candidates=["SKU"], prefer_real=set())
        elif "member" in name.lower():
            df = preprocess_members(df)
            _write_df(conn, df, "members",
                      key_candidates=["Square Customer ID", "Reference ID"],
                      prefer_real=set())

        try:
            os.remove(local)
        except Exception:
            pass

    ensure_indexes()


def init_db_from_drive_once():
    try:
        # 首先确保表存在
        ensure_indexes()

        # 检查是否有数据，如果没有则从Drive导入
        conn = get_db()
        cur = conn.cursor()

        try:
            cur.execute("SELECT COUNT(*) FROM transactions")
            tx_count = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM inventory")
            inv_count = cur.fetchone()[0]

            if tx_count == 0 and inv_count == 0:
                # 只有两个表都为空时才从Drive导入
                ingest_from_drive_all()
        except Exception as e:
            # 如果查询失败，说明表可能不存在，从Drive导入
            ingest_from_drive_all()

    except Exception as e:
        st.sidebar.warning(f"⚠️ Auto-ingest from Drive failed: {e}")
    return True


# --------------- 手动导入（Sidebar 上传） ---------------
def ingest_csv(uploaded_file):
    conn = get_db()

    # 确保表存在
    ensure_indexes()

    filename = uploaded_file.name if hasattr(uploaded_file, "name") else "uploaded.csv"
    st.sidebar.info(f"📂 Importing {filename}")

    try:
        df = pd.read_csv(uploaded_file)
        df = _fix_header(df)

        rows_imported = 0

        if "Net Sales" in df.columns and "Gross Sales" in df.columns:
            df = preprocess_transactions(df)
            _write_df(conn, df, "transactions",
                      key_candidates=["Transaction ID"],
                      prefer_real={"Net Sales", "Gross Sales", "Qty", "Discounts"})
            rows_imported = len(df)
            st.sidebar.success(f"✅ Imported {rows_imported} rows (transactions)")

        elif "SKU" in df.columns or "Stock on Hand" in df.columns or "Categories" in df.columns:
            df = preprocess_inventory(df, filename=filename)
            _write_df(conn, df, "inventory",
                      key_candidates=["SKU"], prefer_real=set())
            rows_imported = len(df)
            st.sidebar.success(f"✅ Imported {rows_imported} rows (inventory)")

        elif "Square Customer ID" in df.columns or "First Name" in df.columns or "member" in filename.lower():
            df = preprocess_members(df)
            _write_df(conn, df, "members",
                      key_candidates=["Square Customer ID", "Reference ID"], prefer_real=set())
            rows_imported = len(df)
            st.sidebar.success(f"✅ Imported {rows_imported} rows (members)")

        else:
            st.sidebar.warning(f"⚠️ Skipped {filename}, schema not recognized")
            return False

        # 上传到 Google Drive
        tmp_path = os.path.join(tempfile.gettempdir(), filename)
        with open(tmp_path, "wb") as f_local:
            if hasattr(uploaded_file, "getbuffer"):
                f_local.write(uploaded_file.getbuffer())
            else:
                f_local.write(uploaded_file.read())
        upload_file_to_drive(tmp_path, filename)

        # 确保索引创建
        ensure_indexes()

        return True

    except Exception as e:
        st.sidebar.error(f"❌ Error importing {filename}: {str(e)}")
        return False
    finally:
        try:
            if 'tmp_path' in locals():
                os.remove(tmp_path)
        except Exception:
            pass


def ingest_excel(uploaded_file):
    conn = get_db()
    ensure_indexes()

    filename = uploaded_file.name if hasattr(uploaded_file, "name") else "uploaded.xlsx"
    st.sidebar.info(f"📂 Importing {filename}")

    try:
        data = uploaded_file.read()
        xls = pd.ExcelFile(BytesIO(data))

        total_rows_imported = 0

        is_catalogue = ("catalogue" in filename.lower())
        # 只处理 Items/首个 sheet（库存）
        if is_catalogue:
            target_sheets = []
            if "Items" in xls.sheet_names:
                target_sheets = ["Items"]
            else:
                # 回退：找第一个 sheet
                target_sheets = [xls.sheet_names[0]]

            inv_frames = []
            for sheet in target_sheets:
                df = pd.read_excel(xls, sheet_name=sheet, header=1)
                df = _fix_header(df)
                if ("SKU" in df.columns) or ("Stock on Hand" in df.columns) or ("Categories" in df.columns):
                    df = preprocess_inventory(df, filename=filename)
                    inv_frames.append(df)

            if inv_frames:
                inv_all = pd.concat(inv_frames, ignore_index=True)
                _write_df(conn, inv_all, "inventory",
                          key_candidates=["SKU"], prefer_real=set())
                total_rows_imported += len(inv_all)
        else:
            # 非 catalogue 的 Excel：保留原逻辑（逐 sheet 导入）
            for sheet in xls.sheet_names:
                header_row = 0
                df = pd.read_excel(xls, sheet_name=sheet, header=header_row)
                df = _fix_header(df)

                if "Net Sales" in df.columns and "Gross Sales" in df.columns:
                    df = preprocess_transactions(df)
                    _write_df(conn, df, "transactions",
                              key_candidates=["Transaction ID"],
                              prefer_real={"Net Sales", "Gross Sales", "Qty", "Discounts"})
                    total_rows_imported += len(df)

                elif "SKU" in df.columns or "Stock on Hand" in df.columns or "Categories" in df.columns:
                    df = preprocess_inventory(df, filename=filename)
                    _write_df(conn, df, "inventory",
                              key_candidates=["SKU"], prefer_real=set())
                    total_rows_imported += len(df)

                elif "Square Customer ID" in df.columns or "First Name" in df.columns or "member" in filename.lower():
                    df = preprocess_members(df)
                    _write_df(conn, df, "members",
                              key_candidates=["Square Customer ID", "Reference ID"], prefer_real=set())
                    total_rows_imported += len(df)

        st.sidebar.success(f"✅ {filename} imported - {total_rows_imported} total rows")

        # 上传到 Drive（保持原逻辑）
        tmp_path = os.path.join(tempfile.gettempdir(), filename)
        with open(tmp_path, "wb") as f_local:
            f_local.write(data)
        upload_file_to_drive(tmp_path, filename)

        ensure_indexes()
        return True

    except Exception as e:
        st.sidebar.error(f"❌ Error importing {filename}: {str(e)}")
        return False
    finally:
        try:
            if 'tmp_path' in locals():
                os.remove(tmp_path)
        except Exception:
            pass


__all__ = [
    "ingest_csv",
    "ingest_excel",
    "ingest_from_drive_all",
    "get_drive",
    "upload_file_to_drive",
    "download_file_from_drive",
    "init_db_from_drive_once",
]