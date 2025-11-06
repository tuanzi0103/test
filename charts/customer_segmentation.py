import streamlit as st

import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
from services.analytics import (
    member_flagged_transactions,
    member_frequency_stats,
    non_member_overview,
    category_counts,
    heatmap_pivot,
    top_categories_for_customer,
    recommend_similar_categories,
    ltv_timeseries_for_customer,
    recommend_bundles_for_customer,
    churn_signals_for_member,
)


def format_phone_number(phone):
    """
    格式化手机号：移除61之前的所有字符，确保以61开头
    """
    if pd.isna(phone) or phone is None:
        return ""

    phone_str = str(phone).strip()

    # 移除所有非数字字符
    digits_only = re.sub(r'\D', '', phone_str)

    # 查找61的位置
    if '61' in digits_only:
        # 找到61第一次出现的位置
        start_index = digits_only.find('61')
        # 返回从61开始的部分
        formatted = digits_only[start_index:]

        # 确保长度合理（手机号通常10-12位）
        if len(formatted) >= 10 and len(formatted) <= 12:
            return formatted
        else:
            # 如果长度不合适，返回原始数字
            return digits_only
    else:
        # 如果没有61，返回原始数字
        return digits_only


def persisting_multiselect(label, options, key, default=None, width_chars=None, format_func=None):
    """
    保持选择状态的多选框函数 - 统一宽度和箭头显示（增强版）
    """
    if key not in st.session_state:
        st.session_state[key] = default or []

    if width_chars is None:
        min_width = 30  # 全局默认 30ch
    else:
        min_width = width_chars

    st.markdown(f"""
    <style>
    /* === 强制覆盖 stMultiSelect 宽度（仅限当前 key） === */
    div[data-testid="stMultiSelect"][data-testid*="{key}"],
    [data-testid*="{key}"][data-testid="stMultiSelect"] {{
        width: {min_width}ch !important;
        min-width: {min_width}ch !important;
        max-width: {min_width}ch !important;
        flex: 0 0 {min_width}ch !important;
        box-sizing: border-box !important;
    }}

    /* === 下拉框主体 === */
    div[data-testid="stMultiSelect"][data-testid*="{key}"] [data-baseweb="select"],
    div[data-testid="stMultiSelect"][data-testid*="{key}"] [data-baseweb="select"] > div {{
        width: 100% !important;
        min-width: 100% !important;
        max-width: 100% !important;
        box-sizing: border-box !important;
    }}

    /* === 输入框 === */
    div[data-testid="stMultiSelect"][data-testid*="{key}"] input {{
        width: 100% !important;
        box-sizing: border-box !important;
    }}

    /* === 下拉菜单 === */
    div[role="listbox"] {{
        width: {min_width}ch !important;
        min-width: {min_width}ch !important;
        max-width: {min_width}ch !important;
        box-sizing: border-box !important;
    }}
    </style>
    """, unsafe_allow_html=True)

    # 确保所有选项都是字符串类型
    options = [str(opt) for opt in options]

    # 确保默认值也是字符串类型
    default_values = [str(val) for val in st.session_state[key]]

    # 创建一个安全的 format_func，确保返回字符串
    def safe_format_func(x):
        result = format_func(x) if format_func else x
        return str(result)

    if format_func:
        return st.multiselect(label, options, default=default_values, key=key, format_func=safe_format_func)
    else:
        return st.multiselect(label, options, default=default_values, key=key)

def is_phone_number(name):
    """
    判断字符串是否为手机号（包含数字和特定字符）
    """
    if pd.isna(name) or name is None:
        return False

    name_str = str(name).strip()

    # 如果字符串只包含数字、空格、括号、加号、连字符，则认为是手机号
    if re.match(r'^[\d\s\(\)\+\-]+$', name_str):
        return True

    # 如果字符串长度在8-15之间且主要包含数字，也认为是手机号
    if 8 <= len(name_str) <= 15 and sum(c.isdigit() for c in name_str) >= 7:
        return True

    return False


def show_customer_segmentation(tx, members):
    # === 全局样式：参考 inventory 的样式设置 ===
    st.markdown("""
    <style>
    /* 去掉标题之间的空白 */
    div.block-container h1, 
    div.block-container h2, 
    div.block-container h3, 
    div.block-container h4,
    div.block-container p {
        margin-top: 0rem !important;
        margin-bottom: 0rem !important;
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
    }

    /* 更强力地压缩 Streamlit 自动插入的 vertical space */
    div.block-container > div {
        margin-top: 0rem !important;
        margin-bottom: 0rem !important;
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
    }

    /* 消除标题和选择框之间空隙 */
    div[data-testid="stVerticalBlock"] > div {
        margin-top: 0rem !important;
        margin-bottom: 0rem !important;
    }

    /* 让多选框列更紧凑 */
    div[data-testid="column"] {
        padding: 0 8px !important;
    }
    /* 让表格文字左对齐 */
    [data-testid="stDataFrame"] table {
        text-align: left !important;
    }
    [data-testid="stDataFrame"] th {
        text-align: left !important;
    }
    [data-testid="stDataFrame"] td {
        text-align: left !important;
    }

    /* 统一多选框和输入框的垂直对齐 */
    div[data-testid="stVerticalBlock"] > div[data-testid="stHorizontalBlock"] {
        align-items: start !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h2 style='font-size:24px; font-weight:700;'>👥 Customer Segmentation & Personalization</h2>",
                unsafe_allow_html=True)

    if tx.empty:
        st.info("No transaction data available.")
        return

    # always use latest uploaded data
    tx = tx.copy()
    members = members.copy()

    # === Prepare Datetime column ===
    tx["Datetime"] = pd.to_datetime(tx.get("Datetime", pd.NaT), errors="coerce")
    today = pd.Timestamp.today().normalize()
    four_weeks_ago = today - pd.Timedelta(weeks=4)
    # ⚠️ 不提前过滤，这样 period 1 可以用到最早的数据
    # tx = tx[(tx["Datetime"] >= four_weeks_ago) & (tx["Datetime"] <= today)]

    # --- 给交易数据打上 is_member 标记
    df = member_flagged_transactions(tx, members)
    # === 新增：统一 Customer Name 与最新 Customer ID ===
    if "Customer Name" in df.columns and "Customer ID" in df.columns and "Datetime" in df.columns:
        # 确保 Datetime 为时间格式
        df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")

        # 找到每个 Customer Name 最近一次交易对应的 Customer ID
        latest_ids = (df.dropna(subset=["Customer Name", "Customer ID", "Datetime"])
                      .sort_values("Datetime")
                      .groupby("Customer Name")
                      .tail(1)[["Customer Name", "Customer ID"]]
                      .drop_duplicates("Customer Name"))

        # 更新 df 中的 Customer ID
        df = df.drop(columns=["Customer ID"]).merge(latest_ids, on="Customer Name", how="left")

    # =========================
    # 👑 前置功能（User Analysis 之前）
    # =========================

    st.markdown("<h3 style='font-size:20px; font-weight:700;'>✨ Overview add-ons</h3>",
                unsafe_allow_html=True)

    # [1] KPI - 参考 Inventory Summary 格式
    net_col = "Net Sales" if "Net Sales" in df.columns else None
    cid_col = "Customer ID" if "Customer ID" in df.columns else None
    avg_spend_member = avg_spend_non_member = None
    if net_col and cid_col and "is_member" in df.columns:
        nets = pd.to_numeric(df[net_col], errors="coerce")
        df_kpi = df.assign(_net=nets)
        avg_spend_member = df_kpi[df_kpi["is_member"]]["_net"].mean()
        avg_spend_non_member = df_kpi[~df_kpi["is_member"]]["_net"].mean()

    # 创建类似 Inventory Summary 格式的数据框
    summary_table_data = {
        'Metric': ['Avg Spend (Enrolled)', 'Avg Spend (Not Enrolled)'],
        'Value': [
            "-" if pd.isna(avg_spend_member) else f"${avg_spend_member:,.2f}",
            "-" if pd.isna(avg_spend_non_member) else f"${avg_spend_non_member:,.2f}"
        ]
    }

    df_summary = pd.DataFrame(summary_table_data)

    # 设置列配置 - 参考 inventory 格式
    column_config = {
        'Metric': st.column_config.Column(width=150),
        'Value': st.column_config.Column(width=50),
    }

    # 显示表格
    st.dataframe(
        df_summary,
        column_config=column_config,
        hide_index=True,
        use_container_width=False
    )

    st.divider()

    # [2] 两个柱状预测 - 放在同一行
    st.markdown("<h3 style='font-size:20px; font-weight:700;'>📊 Customer Behavior Predictions</h3>",
                unsafe_allow_html=True)

    # 使用两列布局将两个预测图表放在同一行
    col1, col2 = st.columns(2)

    time_col = next((c for c in ["Datetime", "Date", "date", "Transaction Time"] if c in df.columns), None)
    if time_col:
        with col1:
            t = pd.to_datetime(df[time_col], errors="coerce")
            day_df = df.assign(_dow=t.dt.day_name())
            dow_counts = day_df.dropna(subset=["_dow"]).groupby("_dow").size().reset_index(
                name="Predicted Transactions")
            cat_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            dow_counts["_dow"] = pd.Categorical(dow_counts["_dow"], categories=cat_order, ordered=True)

            fig_dow = px.bar(
                dow_counts.sort_values("_dow"),
                x="_dow",
                y="Predicted Transactions",
                title="Shopping Days Prediction"
            )
            fig_dow.update_layout(
                width=400,
                height=400,
                xaxis_title=None,  # 去掉横轴标题
                yaxis_title="Predicted Transactions",
                margin=dict(l=40, r=10, t=60, b=30)
            )
            st.plotly_chart(fig_dow, use_container_width=False)

    # 修改：使用分类而不是具体商品名称
    category_col = next((c for c in ["Category", "Item Category", "Product Category"] if c in df.columns), None)
    qty_col = "Qty" if "Qty" in df.columns else None
    if category_col:
        with col2:
            if qty_col:
                top_categories = df.groupby(category_col)[qty_col].sum().reset_index().sort_values(qty_col,
                                                                                                   ascending=False).head(
                    15)
                # 设置柱形图宽度为更紧凑
                fig_categories = px.bar(top_categories, x=category_col, y=qty_col,
                                        title="Top Categories Prediction (Top 15)")
                fig_categories.update_layout(width=400, height=400)  # 设置图表宽度和高度
                st.plotly_chart(fig_categories, use_container_width=False)
            else:
                top_categories = df[category_col].value_counts().reset_index().rename(
                    columns={"index": "Category", category_col: "Count"}).head(15)
                # 设置柱形图宽度为更紧凑
                fig_categories = px.bar(top_categories, x="Category", y="Count",
                                        title="Top Categories Prediction (Top 15)")
                fig_categories.update_layout(width=400, height=400)  # 设置图表宽度和高度
                st.plotly_chart(fig_categories, use_container_width=False)
    else:
        # 如果没有分类列，使用商品名称但只显示大类（通过截取或分组）
        item_col = next((c for c in ["Item", "Item Name", "Variation Name", "SKU Name"] if c in df.columns), None)
        if item_col:
            with col2:
                # 尝试从商品名称中提取分类（取第一个单词或特定分隔符前的部分）
                df_with_category = df.copy()
                # 简单的分类提取：取第一个单词或特定分隔符前的部分
                df_with_category['_category'] = df_with_category[item_col].astype(str).str.split().str[0]

                if qty_col:
                    top_categories = df_with_category.groupby('_category')[qty_col].sum().reset_index().sort_values(
                        qty_col, ascending=False).head(15)
                    fig_categories = px.bar(top_categories, x='_category', y=qty_col,
                                            title="Top Categories Prediction (Top 15)")
                    fig_categories.update_layout(width=400, height=400)
                    st.plotly_chart(fig_categories, use_container_width=False)
                else:
                    top_categories = df_with_category['_category'].value_counts().reset_index().rename(
                        columns={"index": "Category", '_category': "Count"}).head(15)
                    fig_categories = px.bar(top_categories, x="Category", y="Count",
                                            title="Top Categories Prediction (Top 15)")
                    fig_categories.update_layout(width=400, height=400)
                    st.plotly_chart(fig_categories, use_container_width=False)

    st.divider()

    # [3] Top20 churn 风险（基于 Customer Name 计算）
    if time_col and "Customer Name" in df.columns:
        t = pd.to_datetime(df[time_col], errors="coerce")
        df["_ts"] = t

        # === 使用正确的日期范围计算 ===
        today = pd.Timestamp.today().normalize()

        # 第一个期间：从数据的实际第一天到四周前（28天前）
        data_start_date = df["_ts"].min().normalize()  # 使用数据的实际开始日期
        period1_end = today - pd.Timedelta(days=28)  # 四周前

        # 第二个期间：过去四周（今天往前推28天）
        period2_start = today - pd.Timedelta(days=28)
        period2_end = today

        # 检查日期范围是否有效
        if period1_end < data_start_date:
            st.warning(
                f"⚠️ Period 1 end date ({period1_end}) is before data start date ({data_start_date}). Adjusting Period 1 to use available data.")
            # 如果Period 1结束日期在数据开始之前，调整Period 1为数据开始到Period 2开始前一天
            period1_end = period2_start - pd.Timedelta(days=1)
            st.write(f"Adjusted Period 1: {data_start_date} to {period1_end}")

        # === 直接按日期过滤 ===
        base = df.dropna(subset=["Customer Name"])

        # 第一个期间：历史数据（从数据开始到四周前）
        mask_period1 = (base["_ts"] >= data_start_date) & (base["_ts"] <= period1_end)
        period1_data = base[mask_period1]

        # 第二个期间：最近四周
        mask_period2 = (base["_ts"] >= period2_start) & (base["_ts"] <= period2_end)
        period2_data = base[mask_period2]

        # 获取第一个期间的客户（历史常客）
        if not period1_data.empty:
            # 计算历史访问频率（按天去重）
            period1_visits = (period1_data.dropna(subset=["Customer Name", "Transaction ID"])
                              .groupby(["Customer Name", period1_data["_ts"].dt.date])["Transaction ID"]
                              .nunique()
                              .reset_index(name="daily_visits"))

            # === 修改：计算平均每月来访次数（仅对有来访的月份取平均） ===
            period1_visits["_month"] = pd.to_datetime(period1_visits["_ts"]).dt.to_period("M")

            # 每个客户在每个月的访问次数（去重按天或交易）
            monthly_visits = (period1_visits.groupby(["Customer Name", "_month"])
                              ["daily_visits"].sum()
                              .reset_index(name="monthly_visits"))

            # 对每个客户计算平均每月来访次数（仅统计有来访的月份）
            customer_avg_visits = (monthly_visits.groupby("Customer Name")["monthly_visits"]
                                   .mean()
                                   .reset_index(name="Average Visit"))
            customer_avg_visits["Average Visit"] = customer_avg_visits["Average Visit"].round(2)

            # 过滤常客（平均访问次数 >= 2）
            regular_customers = customer_avg_visits[customer_avg_visits["Average Visit"] >= 2]

        else:
            regular_customers = pd.DataFrame(columns=["Customer Name", "Average Visit"])
            st.warning("No data found in Period 1. This might be because the data only started recently.")

        # 获取第二个期间的客户
        if not period2_data.empty:
            period2_customers = period2_data["Customer Name"].drop_duplicates().tolist()

        else:
            period2_customers = []
            st.warning("No data found in Period 2.")

        # 找出流失客户：在第一个期间是常客，但在第二个期间没有出现
        if not regular_customers.empty and period2_customers:
            # 找出在第二个期间没有出现的常客
            lost_customers = regular_customers[~regular_customers["Customer Name"].isin(period2_customers)].copy()

            # 添加 Last Month Visit 列（都为0，因为他们在第二个期间没出现）
            lost_customers["Last Month Visit"] = 0

            # 排序并取前20
            churn_tag_final = lost_customers.sort_values("Average Visit", ascending=False).head(20)
        else:
            churn_tag_final = pd.DataFrame(columns=["Customer Name", "Average Visit", "Last Month Visit"])
            if regular_customers.empty:
                st.info("No regular customers found in historical data.")
            else:
                st.info("No period 2 data to compare against.")

        # 映射 Customer ID 和手机号
        if not churn_tag_final.empty:
            # 获取 Customer ID 映射
            if "Customer ID" in df.columns:
                id_mapping = df[["Customer Name", "Customer ID"]].drop_duplicates().dropna()
                churn_tag_final = churn_tag_final.merge(id_mapping, on="Customer Name", how="left")
            else:
                churn_tag_final["Customer ID"] = ""

            # 映射手机号
            if "Square Customer ID" in members.columns and "Customer ID" in churn_tag_final.columns:
                phones_map = (
                    members.rename(columns={"Square Customer ID": "Customer ID", "Phone Number": "Phone"})
                    [["Customer ID", "Phone"]]
                    .dropna(subset=["Customer ID"])
                    .drop_duplicates("Customer ID")
                )
                phones_map["Customer ID"] = phones_map["Customer ID"].astype(str)
                phones_map["Phone"] = phones_map["Phone"].apply(format_phone_number)

                if "Customer ID" in churn_tag_final.columns:
                    churn_tag_final["Customer ID"] = churn_tag_final["Customer ID"].astype(str)
                    churn_tag_final = churn_tag_final.merge(phones_map, on="Customer ID", how="left")
                else:
                    churn_tag_final["Phone"] = ""
            else:
                churn_tag_final["Phone"] = ""

        st.markdown("<h3 style='font-size:20px; font-weight:700;'>Top 20 Regulars who didn't come last month</h3>",
                    unsafe_allow_html=True)

        # 显示结果
        if not churn_tag_final.empty:
            # === 设置表格列宽配置 ===
            column_config = {
                'Customer Name': st.column_config.Column(width=105),
                'Customer ID': st.column_config.Column(width=100),
                'Phone': st.column_config.Column(width=90),
                'Average Visit': st.column_config.Column(width=90),
                'Last Month Visit': st.column_config.Column(width=110),
            }

            st.dataframe(
                churn_tag_final[["Customer Name", "Customer ID", "Phone",
                                 "Average Visit", "Last Month Visit"]],
                column_config=column_config,
                use_container_width=False
            )
        else:
            st.info("No regular customers found who didn't visit in the last month.")

    st.divider()

    # [4] 姓名/ID 搜索（显示姓名，支持用 ID 搜索）
    options = []
    if "Customer ID" in tx.columns and "Customer Name" in tx.columns:
        # ✅ 仅保留每个 Customer Name 的最新一条 Customer ID
        options = (
            tx.dropna(subset=["Customer ID", "Customer Name", "Datetime"])
            .sort_values("Datetime", ascending=False)
            .drop_duplicates(subset=["Customer Name"])  # 保留每个名字最新的一条记录
            [["Customer ID", "Customer Name"]]
        )

        # 🚩 确保 Customer ID 全部是字符串，避免 multiselect 报错
        options["Customer ID"] = options["Customer ID"].astype(str)
        options = options.to_dict(orient="records")

    # 🔹 使用三列布局缩短下拉框宽度，与 inventory.py 保持一致
    col_search, _ = st.columns([1.2, 5.8])
    with col_search:
        # 创建选项映射
        # ✅ 下拉框只显示用户名，不显示ID
        option_dict = {str(opt["Customer ID"]): str(opt["Customer Name"]) for opt in options}

        # 确保选项是字符串类型
        customer_options = [str(opt["Customer ID"]) for opt in options]

        # 初始化 session state
        if "customer_search_ids" not in st.session_state:
            st.session_state["customer_search_ids"] = []

        # 为分类选择创建表单，避免立即 rerun
        with st.form(key="customer_search_form"):
            sel_ids = st.multiselect(
                "🔎 Search customers",
                options=customer_options,
                default=st.session_state.get("customer_search_ids", []),
                format_func=lambda x: option_dict.get(x, x),
                key="customer_search_widget"
            )

            # 应用按钮
            submitted = st.form_submit_button("Apply", type="primary", use_container_width=True)

            if submitted:
                # 更新 session state
                st.session_state["customer_search_ids"] = sel_ids
                st.rerun()

        # 从 session state 获取最终的选择
        sel_ids = st.session_state.get("customer_search_ids", [])

        # 显示当前选择状态
        if sel_ids:
            st.caption(f"✅ Selected: {len(sel_ids)} customers")
        else:
            st.caption("ℹ️ No customers selected")

    if sel_ids:
        # === 修复：兼容 Customer ID 变更或为空的情况 ===
        # 根据选中 ID 找出对应的 Customer Name
        sel_names = tx[tx["Customer ID"].astype(str).isin(sel_ids)]["Customer Name"].dropna().unique().tolist()

        # 匹配逻辑：Customer ID 或 Customer Name 任一符合都保留
        chosen = tx[
            tx["Customer ID"].astype(str).isin(sel_ids) |
            tx["Customer Name"].isin(sel_names)
            ]

        st.markdown("<h3 style='font-size:20px; font-weight:700;'>All transactions for selected customers</h3>",
                    unsafe_allow_html=True)

        column_config = {
            "Datetime": st.column_config.Column(width=120),
            "Customer Name": st.column_config.Column(width=120),
            "Customer ID": st.column_config.Column(width=140),
            "Category": st.column_config.Column(width=140),
            "Item": st.column_config.Column(width=250),
            "Qty": st.column_config.Column(width=40),
            "Net Sales": st.column_config.Column(width=80),
        }

        # ✅ 仅显示指定列（按顺序）
        display_cols = ["Datetime", "Customer Name", "Category", "Item", "Qty", "Net Sales"]
        existing_cols = [c for c in display_cols if c in chosen.columns]

        if "Datetime" in chosen.columns:
            chosen = chosen.sort_values("Datetime", ascending=False)

        st.dataframe(
            chosen[existing_cols],
            column_config=column_config,
            use_container_width=False,  # ✅ 关闭容器自适应，列宽才生效
            hide_index=True
        )

        if qty_col:
            # 使用具体的 Item 而不是 Category
            item_col_display = next(
                (c for c in ["Item", "Item Name", "Variation Name", "SKU Name"] if c in chosen.columns), None)

            if item_col_display:
                top5 = (chosen.groupby(["Customer ID", "Customer Name", item_col_display])[qty_col].sum()
                        .reset_index()
                        .sort_values(["Customer Name", qty_col], ascending=[True, False])
                        .groupby("Customer ID").head(5))

                st.markdown(
                    "<h3 style='font-size:20px; font-weight:700;'>Frequently purchased categories (Top 5 / customer)</h3>",
                    unsafe_allow_html=True)

                column_config = {
                    'Customer Name': st.column_config.Column(width=110),
                    item_col_display: st.column_config.Column(width=250),  # 移除 title 参数
                    qty_col: st.column_config.Column(width=40),
                }

                # 同时修改显示的列，去掉 Customer ID，并重命名列标题
                display_df = top5[["Customer Name", item_col_display, qty_col]].rename(
                    columns={item_col_display: "Item"}
                )
                st.dataframe(display_df, column_config=column_config, use_container_width=False)

            else:
                # 如果没有分类列，使用商品名称但显示为分类
                item_col_display = next(
                    (c for c in ["Item", "Item Name", "Variation Name", "SKU Name"] if c in chosen.columns), None)
                if item_col_display:
                    # 从商品名称中提取分类
                    chosen_with_category = chosen.copy()
                    chosen_with_category['_category'] = \
                        chosen_with_category[item_col_display].astype(str).str.split().str[0]

                    top5 = (chosen_with_category.groupby(["Customer ID", "Customer Name", '_category'])[qty_col].sum()
                            .reset_index()
                            .sort_values(["Customer Name", qty_col], ascending=[True, False])
                            .groupby("Customer ID").head(5))

                    st.markdown(
                        "<h3 style='font-size:20px; font-weight:700;'>Frequently purchased categories (Top 5 / customer)</h3>",
                        unsafe_allow_html=True)

                    column_config = {
                        'Customer Name': st.column_config.Column(width=110),
                        '_category': st.column_config.Column(width=250),  # 移除 title 参数
                        qty_col: st.column_config.Column(width=40),
                    }

                    # 去掉 Customer ID 列，并重命名列标题
                    display_df = top5[["Customer Name", "_category", qty_col]].rename(
                        columns={"_category": "Item"}
                    )
                    st.dataframe(display_df, column_config=column_config, use_container_width=False)
    st.divider()

    # [5] Heatmap 可切换
    st.markdown("<h3 style='font-size:20px; font-weight:700;'>Heatmap (selectable metric)</h3>",
                unsafe_allow_html=True)

    # 🔹 使用三列布局缩短下拉框宽度，与 inventory.py 保持一致
    col_metric, _ = st.columns([1, 6])
    with col_metric:
        # === 修改：设置选择框宽度 ===
        st.markdown("""
        <style>
        div[data-testid*="stSelectbox"][aria-label="Metric"],
        div[data-testid*="stSelectbox"][data-baseweb="select"][aria-label="Metric"] {
            width: 15ch !important;
            min-width: 15ch !important;
            max-width: 15ch !important;
        }
        </style>
        """, unsafe_allow_html=True)

        metric = st.selectbox("Metric", ["net sales", "number of transactions"], index=0, key="heatmap_metric")

    if time_col:
        t = pd.to_datetime(df[time_col], errors="coerce")
        base = df.assign(_date=t)
        base["_hour"] = base["_date"].dt.hour
        base["_dow"] = base["_date"].dt.day_name()
        if metric == "net sales" and net_col:
            agg = base.groupby(["_dow", "_hour"])[net_col].sum().reset_index(name="value")
        else:
            txn_col2 = "Transaction ID" if "Transaction ID" in base.columns else None
            if txn_col2:
                agg = base.groupby(["_dow", "_hour"])[txn_col2].nunique().reset_index(name="value")
            else:
                agg = base.groupby(["_dow", "_hour"]).size().reset_index(name="value")
        pv = agg.pivot(index="_dow", columns="_hour", values="value").fillna(0)

        # === 修改：设置热力图宽度 ===
        fig_heatmap = px.imshow(pv, aspect="auto", title=f"Heatmap by {metric.title()} (Hour x Day)")
        fig_heatmap.update_layout(width=600)  # 设置图表宽度
        st.plotly_chart(fig_heatmap, use_container_width=False)