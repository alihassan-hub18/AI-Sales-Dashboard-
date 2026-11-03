import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(
    page_title="AI Sales Dashboard",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #172554 55%, #111827 100%);
        color: #f8fafc;
    }
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #38bdf8, #a78bfa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle {
        text-align: center;
        color: #cbd5e1;
        margin-bottom: 1.5rem;
    }
    div[data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.75);
        border: 1px solid rgba(148, 163, 184, 0.25);
        padding: 15px;
        border-radius: 14px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📊 AI-Powered Sales Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Explore sales, discover trends, and predict sales with machine learning.</div>', unsafe_allow_html=True)

st.sidebar.header("⚙️ Dashboard Controls")
uploaded_file = st.sidebar.file_uploader("Upload your sales CSV", type=["csv"])
use_sample = st.sidebar.checkbox("Use sample sales data", value=True)

def make_sample_data():
    rng = np.random.default_rng(42)
    n = 240
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    categories = ["Technology", "Furniture", "Office Supplies"]
    regions = ["North", "South", "East", "West"]
    products = ["Laptop", "Chair", "Desk", "Printer", "Phone", "Notebook"]
    quantity = rng.integers(1, 9, n)
    price = rng.choice([15, 25, 50, 120, 300, 800], n)
    discount = rng.choice([0, 0.05, 0.10, 0.15, 0.20], n)
    sales = quantity * price * (1 - discount)
    cost = sales * rng.uniform(0.45, 0.78, n)
    return pd.DataFrame({
        "Order Date": dates,
        "Order ID": [f"ORD-{i+1:04d}" for i in range(n)],
        "Product": rng.choice(products, n),
        "Category": rng.choice(categories, n),
        "Region": rng.choice(regions, n),
        "Quantity": quantity,
        "Sales": np.round(sales, 2),
        "Profit": np.round(sales - cost, 2),
        "Discount": discount
    })

def normalize_columns(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    lower_map = {c.lower().strip(): c for c in df.columns}

    aliases = {
        "date": ["date", "order date", "sales date", "transaction date"],
        "sales": ["sales", "revenue", "total sales", "amount", "sales amount"],
        "profit": ["profit", "net profit", "gross profit"],
        "quantity": ["quantity", "qty", "units", "units sold"],
        "category": ["category", "product category", "subcategory"],
        "region": ["region", "area", "territory", "market"],
        "product": ["product", "product name", "item", "item name"],
        "order_id": ["order id", "order_id", "transaction id", "invoice id"]
    }
    rename = {}
    for standard, names in aliases.items():
        for name in names:
            if name in lower_map:
                rename[lower_map[name]] = standard.title().replace("_", " ") if standard != "order_id" else "Order ID"
                break
    df = df.rename(columns=rename)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    elif "Order Date" in df.columns:
        df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")

    for col in ["Sales", "Profit", "Quantity", "Discount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")
    return df

if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
        df = normalize_columns(raw_df)
        st.sidebar.success("CSV uploaded successfully.")
    except Exception as e:
        st.error(f"Could not read this CSV file: {e}")
        st.stop()
elif use_sample:
    df = normalize_columns(make_sample_data())
    st.sidebar.info("Showing generated sample data. Upload your own CSV to analyze it.")
else:
    st.info("Upload a sales CSV file or enable 'Use sample sales data' in the sidebar.")
    st.stop()

if df.empty:
    st.error("The dataset is empty. Please upload a CSV with sales records.")
    st.stop()

# Required sales column
if "Sales" not in df.columns:
    st.error("A sales column was not found. Please include a column named Sales, Revenue, Amount, or Total Sales.")
    st.write("Detected columns:", list(df.columns))
    st.stop()

df = df.dropna(subset=["Sales"]).copy()
if df.empty:
    st.error("No valid numeric sales values were found in the CSV.")
    st.stop()

# Date handling
date_col = next((c for c in ["Order Date", "Date"] if c in df.columns), None)
if date_col:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

# Filters
st.sidebar.subheader("🔎 Filters")
filtered_df = df.copy()

if "Region" in filtered_df.columns:
    regions = sorted(filtered_df["Region"].dropna().astype(str).unique())
    selected_regions = st.sidebar.multiselect("Region", regions, default=regions)
    if selected_regions:
        filtered_df = filtered_df[filtered_df["Region"].astype(str).isin(selected_regions)]

if "Category" in filtered_df.columns:
    categories = sorted(filtered_df["Category"].dropna().astype(str).unique())
    selected_categories = st.sidebar.multiselect("Category", categories, default=categories)
    if selected_categories:
        filtered_df = filtered_df[filtered_df["Category"].astype(str).isin(selected_categories)]

if filtered_df.empty:
    st.warning("No rows match your current filters. Select at least one region/category.")
    st.stop()

total_sales = filtered_df["Sales"].sum()
total_orders = filtered_df["Order ID"].nunique() if "Order ID" in filtered_df.columns else len(filtered_df)
average_sales = filtered_df["Sales"].mean()
total_profit = filtered_df["Profit"].sum() if "Profit" in filtered_df.columns else None

m1, m2, m3, m4 = st.columns(4)
m1.metric("💰 Total Sales", f"{total_sales:,.2f}")
m2.metric("🧾 Total Orders / Rows", f"{total_orders:,}")
m3.metric("📈 Average Sale", f"{average_sales:,.2f}")
m4.metric("💵 Total Profit", f"{total_profit:,.2f}" if total_profit is not None else "Not available")

st.markdown("---")
tab_overview, tab_charts, tab_ml, tab_data = st.tabs(
    ["📌 Overview", "📊 Visualizations", "🤖 Sales Prediction", "📄 Data"]
)

with tab_overview:
    st.subheader("Sales Overview")
    left, right = st.columns(2)

    with left:
        if date_col and filtered_df[date_col].notna().any():
            time_df = filtered_df.dropna(subset=[date_col]).copy()
            time_df["Month"] = time_df[date_col].dt.to_period("M").astype(str)
            monthly = time_df.groupby("Month", as_index=False)["Sales"].sum()
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(monthly["Month"], monthly["Sales"], marker="o")
            ax.set_title("Monthly Sales Trend")
            ax.set_xlabel("Month")
            ax.set_ylabel("Sales")
            ax.tick_params(axis="x", rotation=45)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Add a Date or Order Date column to see monthly sales trends.")

    with right:
        if "Category" in filtered_df.columns:
            category_sales = filtered_df.groupby("Category", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False)
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(data=category_sales, x="Sales", y="Category", ax=ax)
            ax.set_title("Sales by Category")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Add a Category column to compare category sales.")

    st.subheader("🧠 Automatic Insights")
    best_row = None
    if "Category" in filtered_df.columns:
        best_row = filtered_df.groupby("Category")["Sales"].sum().idxmax()
        st.write(f"• Highest-sales category: **{best_row}**.")
    if "Region" in filtered_df.columns:
        best_region = filtered_df.groupby("Region")["Sales"].sum().idxmax()
        st.write(f"• Highest-sales region: **{best_region}**.")
    if date_col and filtered_df[date_col].notna().any():
        valid_dates = filtered_df.dropna(subset=[date_col])
        st.write(f"• Date range: **{valid_dates[date_col].min().date()}** to **{valid_dates[date_col].max().date()}**.")
    st.write(f"• Current filtered dataset contains **{len(filtered_df):,} rows** and total sales of **{total_sales:,.2f}**.")

with tab_charts:
    st.subheader("📊 Sales Data Visualizations")
    c1, c2 = st.columns(2)

    with c1:
        if "Region" in filtered_df.columns:
            region_sales = filtered_df.groupby("Region", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False)
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.barplot(data=region_sales, x="Region", y="Sales", ax=ax)
            ax.set_title("Sales by Region")
            ax.tick_params(axis="x", rotation=25)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    with c2:
        if "Profit" in filtered_df.columns:
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.histplot(data=filtered_df, x="Profit", kde=True, ax=ax)
            ax.set_title("Profit Distribution")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Add a Profit column to view profit distribution.")

    numeric_cols = filtered_df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) >= 2:
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(filtered_df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title("Numeric Feature Correlations")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("At least two numeric columns are needed for a correlation heatmap.")

with tab_ml:
    st.subheader("🤖 Machine Learning: Sales Prediction")
    st.write("This demo trains a Random Forest regression model to estimate sales from available columns. Results depend on the quality and size of your data.")

    target_col = "Sales"
    ml_df = filtered_df.copy().dropna(subset=[target_col])
    excluded = {target_col}
    if "Profit" in ml_df.columns:
        excluded.add("Profit")  # avoids using a value that may be calculated from sales
    if "Order ID" in ml_df.columns:
        excluded.add("Order ID")

    # Derive date features and remove raw date columns
    for c in list(ml_df.columns):
        if c in ["Date", "Order Date"]:
            parsed = pd.to_datetime(ml_df[c], errors="coerce")
            ml_df[c + " Year"] = parsed.dt.year
            ml_df[c + " Month"] = parsed.dt.month
            ml_df[c + " Day"] = parsed.dt.day
            excluded.add(c)

    feature_cols = [c for c in ml_df.columns if c not in excluded and ml_df[c].notna().any()]
    if len(ml_df) < 10:
        st.warning("At least 10 valid rows are recommended to train and evaluate the demo model.")
    elif not feature_cols:
        st.warning("The dataset needs other useful columns, such as Quantity, Category, Region, Product, or Date, in addition to Sales.")
    else:
        model_df = ml_df[feature_cols + [target_col]].copy()
        model_df = model_df.dropna(subset=[target_col])
        for c in feature_cols:
            if model_df[c].dtype == "object":
                model_df[c] = model_df[c].fillna("Unknown").astype(str)
            else:
                model_df[c] = pd.to_numeric(model_df[c], errors="coerce")
                model_df[c] = model_df[c].fillna(model_df[c].median())

        categorical = [c for c in feature_cols if model_df[c].dtype == "object"]
        numeric = [c for c in feature_cols if c not in categorical]
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
                ("num", "passthrough", numeric)
            ],
            remainder="drop"
        )
        model = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(n_estimators=150, random_state=42))
        ])

        if st.button("Train Model & Evaluate"):
            X = model_df[feature_cols]
            y = model_df[target_col]
            test_size = 0.2 if len(model_df) >= 20 else 0.3
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions) if len(y_test) > 1 else float("nan")
            a, b = st.columns(2)
            a.metric("Mean Absolute Error", f"{mae:,.2f}")
            b.metric("R² Score", f"{r2:.3f}" if np.isfinite(r2) else "N/A")
            comparison = pd.DataFrame({"Actual Sales": y_test.values, "Predicted Sales": predictions})
            st.dataframe(comparison, use_container_width=True)
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.scatter(y_test, predictions)
            low = min(float(np.min(y_test)), float(np.min(predictions)))
            high = max(float(np.max(y_test)), float(np.max(predictions)))
            ax.plot([low, high], [low, high], linestyle="--")
            ax.set_xlabel("Actual Sales")
            ax.set_ylabel("Predicted Sales")
            ax.set_title("Actual vs Predicted Sales")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.caption("This is a learning/demo model, not a guarantee of future sales. For real forecasting, use chronological validation and time-series features.")
        st.caption("Features used: " + ", ".join(feature_cols))

with tab_data:
    st.subheader("📄 Filtered Sales Data")
    st.dataframe(filtered_df, use_container_width=True)
    csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Filtered Data as CSV",
        data=csv_bytes,
        file_name="filtered_sales_data.csv",
        mime="text/csv"
    )

st.markdown("---")
st.caption("Built with Streamlit, Pandas, Matplotlib, Seaborn, and Scikit-learn.")
