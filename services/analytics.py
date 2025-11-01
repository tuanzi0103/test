# services/analytics.py
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from datetime import timedelta
from services.db import get_db
import pandas as pd



# === 工具函数 ===
def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.drop_duplicates()
    df = df.dropna(how="all")
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df


def _to_numeric(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(r"[^0-9\.\-]", "", regex=True)
        .replace("", np.nan)
        .astype(float)
    )


# === 数据加载 ===
def load_transactions(db, days=365, time_from=None, time_to=None):
    if time_from and time_to:
        start, end = pd.to_datetime(time_from), pd.to_datetime(time_to)
    else:
        end = pd.Timestamp.today()
        start = end - pd.Timedelta(days=days)

    # 转成 SQLite 可识别的字符串
    start_str = start.strftime("%Y-%m-%d %H:%M:%S")
    end_str = end.strftime("%Y-%m-%d %H:%M:%S")

    query = """
        SELECT Datetime, Category, Item, Qty, [Net Sales], [Gross Sales],
               Discounts, [Customer ID], [Transaction ID]
        FROM transactions
        WHERE Datetime BETWEEN ? AND ?
    """
    df = pd.read_sql(query, db, params=[start_str, end_str])
    if not df.empty:
        df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    return df


def load_inventory(db) -> pd.DataFrame:
    df = pd.read_sql("SELECT * FROM inventory", db)
    return _clean_df(df)


def load_members(db) -> pd.DataFrame:
    df = pd.read_sql("SELECT * FROM members", db)
    return _clean_df(df)


def compute_inventory_profit(df: pd.DataFrame) -> pd.DataFrame:
    """
    修改后的inventory value计算公式：
    - Tax - GST (10%)列如果是N: inventory value = Current Quantity Vie Market & Bar * Default Unit Cost
    - Tax - GST (10%)列如果是Y: inventory value = Current Quantity Vie Market & Bar * (Default Unit Cost/11*10)
    - 过滤掉Current Quantity Vie Market & Bar或者Default Unit Cost为空的行
    """
    if df is None or df.empty:
        return df

    df = df.copy()

    for col in ["Tax - GST (10%)", "Price", "Current Quantity Vie Market & Bar", "Default Unit Cost"]:
        if col not in df.columns:
            df[col] = np.nan

    # 过滤掉空值行
    mask = (~df["Current Quantity Vie Market & Bar"].isna()) & (~df["Default Unit Cost"].isna())
    df = df[mask].copy()

    if df.empty:
        return df

    price = _to_numeric(df["Price"])
    qty = _to_numeric(df["Current Quantity Vie Market & Bar"])
    unit_cost = _to_numeric(df["Default Unit Cost"])
    tax_flag = df["Tax - GST (10%)"].astype(str)

    # 计算 retail_total
    retail_total = pd.Series(0.0, index=df.index)
    retail_total.loc[tax_flag.eq("N")] = (price * qty).loc[tax_flag.eq("N")]
    retail_total.loc[tax_flag.eq("Y")] = ((price / 11.0 * 10.0) * qty).loc[tax_flag.eq("Y")]

    # 修改：计算 inventory_value
    inventory_value = pd.Series(0.0, index=df.index)
    inventory_value.loc[tax_flag.eq("N")] = (unit_cost * qty).loc[tax_flag.eq("N")]
    inventory_value.loc[tax_flag.eq("Y")] = ((unit_cost / 11.0 * 10.0) * qty).loc[tax_flag.eq("Y")]

    profit = retail_total - inventory_value

    df["retail_total"] = retail_total
    df["inventory_value"] = inventory_value
    df["profit"] = profit

    return df


def load_all(db=None, time_from=None, time_to=None, days=None):
    conn = db or get_db()

    tx = pd.read_sql("SELECT * FROM transactions", conn)
    inv = pd.read_sql("SELECT * FROM inventory", conn)

    try:
        mem = pd.read_sql("SELECT * FROM members", conn)
    except Exception:
        mem = pd.DataFrame()

    # ✅ 每次都重新计算 inventory_value / profit，保证口径一致
    if not inv.empty:
        inv = compute_inventory_profit(inv)

    return tx, mem, inv


# === 日报表 ===
def daily_summary(transactions: pd.DataFrame) -> pd.DataFrame:
    if transactions.empty:
        return pd.DataFrame()
    transactions["date"] = pd.to_datetime(transactions["Datetime"], errors="coerce").dt.date

    # 👉 确保关键列为数值，避免 groupby 后求和/均值时出错
    for col in ["Net Sales", "Gross Sales", "Qty"]:
        if col in transactions.columns:
            transactions[col] = (
                transactions[col]
                .astype(str)
                .str.replace(r"[^0-9\.\-]", "", regex=True)
                .replace("", pd.NA)
            )
            transactions[col] = pd.to_numeric(transactions[col], errors="coerce")

    summary = (
        transactions.groupby("date")
        .agg(
            net_sales=("Net Sales", "sum"),
            transactions=("Datetime", "count"),
            avg_txn=("Net Sales", "mean"),
            gross=("Gross Sales", "sum"),
            qty=("Qty", "sum"),
        )
        .reset_index()
    )
    summary["profit"] = summary["gross"] - summary["net_sales"]
    return summary


# === 销售预测 ===
def forecast_sales(transactions: pd.DataFrame, periods: int = 30) -> pd.DataFrame:
    if transactions.empty:
        return pd.DataFrame()
    transactions["date"] = pd.to_datetime(transactions["Datetime"]).dt.date
    daily_sales = transactions.groupby("date")["Net Sales"].sum()
    if len(daily_sales) < 10:
        return pd.DataFrame()
    model = ExponentialSmoothing(daily_sales, trend="add", seasonal=None)
    fit = model.fit()
    forecast = fit.forecast(periods)
    return pd.DataFrame({
        "date": pd.date_range(start=daily_sales.index[-1] + timedelta(days=1), periods=periods),
        "forecast": forecast.values
    })


# === 高消费客户 ===
def forecast_top_consumers(transactions: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    if transactions.empty or "Customer ID" not in transactions.columns:
        return pd.DataFrame()
    return (
        transactions.groupby("Customer ID")["Net Sales"]
        .sum()
        .reset_index()
        .sort_values("Net Sales", ascending=False)
        .head(top_n)
    )


# === SKU 消耗时序 ===
def sku_consumption_timeseries(transactions: pd.DataFrame, sku: str) -> pd.DataFrame:
    if transactions.empty or "Item" not in transactions.columns:
        return pd.DataFrame()
    df = transactions[transactions["Item"] == sku].copy()
    if df.empty:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["Datetime"]).dt.date
    return df.groupby("date")["Qty"].sum().reset_index()


# === 会员相关分析 ===
def member_flagged_transactions(transactions: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    if transactions.empty or members.empty:
        return transactions
    member_ids = set(members["Square Customer ID"].unique())
    transactions = transactions.copy()
    transactions["is_member"] = transactions["Customer ID"].apply(lambda x: x in member_ids)
    return transactions


def member_frequency_stats(transactions: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    if transactions.empty or members.empty:
        return pd.DataFrame()
    df = transactions.copy()
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    df = df.dropna(subset=["Datetime"])
    stats = (
        df.groupby("Customer ID")["Datetime"]
        .agg(["count", "min", "max"])
        .reset_index()
        .rename(columns={"count": "txn_count", "min": "first_txn", "max": "last_txn"})
    )
    # ✅ 用新的列名来计算
    stats["days_active"] = (stats["last_txn"] - stats["first_txn"]).dt.days.clip(lower=1)
    stats["avg_days_between"] = stats["days_active"] / stats["txn_count"]
    return stats


def non_member_overview(transactions: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    if transactions.empty:
        return pd.DataFrame()
    member_ids = set(members["Square Customer ID"].unique()) if not members.empty else set()
    df = transactions[~transactions["Customer ID"].isin(member_ids)].copy()
    return df.groupby("Customer ID")["Net Sales"].sum().reset_index()


# === 分类与推荐分析 ===
def category_counts(transactions: pd.DataFrame) -> pd.DataFrame:
    if transactions.empty or "Category" not in transactions.columns:
        return pd.DataFrame()
    return transactions["Category"].value_counts().reset_index().rename(
        columns={"index": "Category", "Category": "count"})


def heatmap_pivot(transactions: pd.DataFrame) -> pd.DataFrame:
    if transactions.empty or "Category" not in transactions.columns:
        return pd.DataFrame()
    return pd.pivot_table(
        transactions, values="Net Sales", index="Customer ID", columns="Category", aggfunc="sum", fill_value=0
    )


def top_categories_for_customer(transactions: pd.DataFrame, customer_id: str, top_n: int = 3) -> pd.DataFrame:
    df = transactions[transactions["Customer ID"] == customer_id]
    if df.empty:
        return pd.DataFrame()
    return (
        df.groupby("Category")["Net Sales"]
        .sum()
        .reset_index()
        .sort_values("Net Sales", ascending=False)
        .head(top_n)
    )


def recommend_similar_categories(transactions: pd.DataFrame, category: str, top_n: int = 3) -> pd.DataFrame:
    if transactions.empty or "Category" not in transactions.columns:
        return pd.DataFrame()
    other_cats = transactions["Category"].value_counts().reset_index()
    other_cats = other_cats[other_cats["index"] != category]
    return other_cats.head(top_n)


def ltv_timeseries_for_customer(transactions: pd.DataFrame, customer_id: str) -> pd.DataFrame:
    df = transactions[transactions["Customer ID"] == customer_id]
    if df.empty:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["Datetime"]).dt.date
    return df.groupby("date")["Net Sales"].sum().cumsum().reset_index()


def recommend_bundles_for_customer(transactions: pd.DataFrame, customer_id: str, top_n: int = 3) -> pd.DataFrame:
    df = transactions[transactions["Customer ID"] == customer_id]
    if df.empty or "Item" not in df.columns:
        return pd.DataFrame()
    return df["Item"].value_counts().reset_index().head(top_n)


def churn_signals_for_member(transactions: pd.DataFrame, members: pd.DataFrame,
                             days_threshold: int = 30) -> pd.DataFrame:
    if transactions.empty or members.empty:
        return pd.DataFrame()
    df = transactions[transactions["Customer ID"].isin(members["Square Customer ID"].unique())]
    if df.empty:
        return pd.DataFrame()
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    last_seen = df.groupby("Customer ID")["Datetime"].max().reset_index()
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=days_threshold)
    last_seen["churn_flag"] = last_seen["Datetime"] < cutoff
    return last_seen