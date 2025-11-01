# services/db.py
import sqlite3
import os
from contextlib import contextmanager

# 数据库文件路径
DB_PATH = 'manlyfarm.db'


def get_db():
    """获取数据库连接"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # 允许以字典方式访问行
    return conn


@contextmanager
def db_connection():
    """数据库连接上下文管理器"""
    conn = get_db()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_database():
    """初始化数据库表结构"""
    conn = get_db()
    cur = conn.cursor()

    # 创建 transactions 表
    cur.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            Datetime TEXT,
            Category TEXT,
            Item TEXT,
            Qty REAL,
            [Net Sales] REAL,
            [Gross Sales] REAL,
            Discounts REAL,
            [Customer ID] TEXT,
            [Transaction ID] TEXT,
            Tax TEXT,
            [Card Brand] TEXT,
            [PAN Suffix] TEXT,
            [Date] TEXT,
            [Time] TEXT,
            [Time Zone] TEXT,
            [Modifiers Applied] TEXT
        )
    """)

    # 创建 inventory 表
    cur.execute("""
        CREATE TABLE IF NOT EXISTS inventory (
            [Product ID] TEXT,
            [Product Name] TEXT,
            SKU TEXT,
            Categories TEXT,
            Price REAL,
            [Tax - GST (10%)] TEXT,
            [Current Quantity Vie Market & Bar] REAL,
            [Default Unit Cost] REAL,
            Unit TEXT,
            source_date TEXT,
            [Stock on Hand] REAL
        )
    """)

    # 创建 members 表
    cur.execute("""
        CREATE TABLE IF NOT EXISTS members (
            [Square Customer ID] TEXT,
            [First Name] TEXT,
            [Last Name] TEXT,
            [Email Address] TEXT,
            [Phone Number] TEXT,
            [Creation Date] TEXT,
            [Customer Note] TEXT,
            [Reference ID] TEXT
        )
    """)

    # 创建 units 表
    cur.execute("""
        CREATE TABLE IF NOT EXISTS units (
            name TEXT UNIQUE
        )
    """)

    # 创建索引以提高查询性能
    cur.execute('CREATE INDEX IF NOT EXISTS idx_txn_datetime ON transactions(Datetime)')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_txn_id ON transactions([Transaction ID])')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_member_square ON members([Square Customer ID])')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_member_ref ON members([Reference ID])')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_inv_sku ON inventory(SKU)')
    cur.execute('CREATE INDEX IF NOT EXISTS idx_inv_categories ON inventory(Categories)')

    conn.commit()
    conn.close()


def table_exists(table_name):
    """检查表是否存在"""
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        return cur.fetchone() is not None
    finally:
        conn.close()


def get_table_columns(table_name):
    """获取表的列信息"""
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(f"PRAGMA table_info({table_name})")
        return [row[1] for row in cur.fetchall()]
    except Exception:
        return []
    finally:
        conn.close()


def execute_query(query, params=None):
    """执行查询并返回结果"""
    conn = get_db()
    try:
        cur = conn.cursor()
        if params:
            cur.execute(query, params)
        else:
            cur.execute(query)

        if query.strip().upper().startswith('SELECT'):
            return cur.fetchall()
        else:
            conn.commit()
            return cur.rowcount
    finally:
        conn.close()


def get_table_row_count(table_name):
    """获取表的行数"""
    conn = get_db()
    try:
        cur = conn.cursor()
        cur.execute(f"SELECT COUNT(*) FROM {table_name}")
        return cur.fetchone()[0]
    except Exception:
        return 0
    finally:
        conn.close()


# 导出函数
__all__ = [
    'get_db',
    'db_connection',
    'init_database',
    'table_exists',
    'get_table_columns',
    'execute_query',
    'get_table_row_count'
]