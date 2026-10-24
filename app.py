import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

# ---------------- PAGE CONFIGURATION ----------------
st.set_page_config(
    page_title="SaaS Churn Intelligence",
    page_icon="📊",
    layout="wide"
)

# Use the folder containing this app.py file.
# Do not use Colab's /content path on Streamlit Cloud.
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "saas_churn_dataset.csv"
MODEL_FILE = BASE_DIR / "saas_churn_model.pkl"

# ---------------- CUSTOM DESIGN ----------------
st.markdown("""
<style>
.stApp {
    background-color: #F3F6FC;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #101D38, #203F70);
}
[data-testid="stSidebar"] * {
    color: white;
}
.block-container {
    padding-top: 2rem;
}
.hero {
    background: linear-gradient(120deg, #142746, #3266B2);
    padding: 28px;
    border-radius: 16px;
    color: white;
    margin-bottom: 24px;
}
.hero h1 {
    color: white;
    font-size: 30px;
}
.hero p {
    color: #E0EAFE;
}
div[data-testid="stMetric"] {
    background: white;
    padding: 18px;
    border: 1px solid #E1E8F0;
    border-radius: 12px;
}
div[data-testid="stMetricValue"] {
    color: #18345C;
}
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_FILE)


# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_FILE)


if not DATA_FILE.exists():
    st.error("Dataset file not found: saas_churn_dataset.csv")
    st.info(
        "Make sure saas_churn_dataset.csv is uploaded "
        "to the same GitHub repository as app.py."
    )
    st.stop()

try:
    df = load_data()
except Exception as e:
    st.error(f"Unable to load dataset: {e}")
    st.stop()


# ---------------- VALIDATE DATA ----------------
required_columns = [
    "CustomerID",
    "TenureMonths",
    "MonthlyCharges",
    "UsageHours",
    "SupportTickets",
    "PaymentDelays",
    "Churn"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error(f"Missing dataset columns: {missing_columns}")
    st.stop()

if df.empty:
    st.error("The dataset is empty.")
    st.stop()

try:
    df["Churn"] = pd.to_numeric(df["Churn"], errors="raise")

    if not df["Churn"].isin([0, 1]).all():
        st.error("Churn column must contain only 0 and 1.")
        st.stop()

    df["Churn"] = df["Churn"].astype(int)

except Exception as e:
    st.error(f"Invalid Churn column: {e}")
    st.stop()


# ---------------- SIDEBAR ----------------
st.sidebar.title("📊 SaaS Intelligence")
st.sidebar.caption("SUBSCRIPTION ANALYTICS")

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Customer Data", "Churn Prediction"]
)

st.sidebar.divider()
st.sidebar.caption("Customer Retention Analytics")


# ---------------- BUSINESS METRICS ----------------
total = len(df)
churned = int(df["Churn"].sum())
active = total - churned
churn_rate = (churned / total * 100) if total else 0

st.markdown("""
<div class="hero">
    <h1>SaaS Subscription Churn Intelligence</h1>
    <p>
        Understand customer retention, explore customer data,
        and estimate subscription churn risk.
    </p>
</div>
""", unsafe_allow_html=True)


# ==================================================
# DASHBOARD
# ==================================================
if page == "Dashboard":

    st.subheader("Business Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Customers", f"{total:,}")
    col2.metric("Active Customers", f"{active:,}")
    col3.metric("Churned Customers", f"{churned:,}")
    col4.metric("Churn Rate", f"{churn_rate:.1f}%")

    left, right = st.columns(2)

    with left:
        st.subheader("Customer Status")

        fig, ax = plt.subplots(figsize=(6, 4))

        bars = ax.bar(
            ["Active", "Churned"],
            [active, churned],
            color=["#16A085", "#E76F51"],
            width=0.55
        )

        ax.set_ylabel("Number of Customers")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.bar_label(bars, padding=4)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.subheader("Customer Distribution")

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.pie(
            [active, churned],
            labels=["Active", "Churned"],
            autopct="%1.1f%%",
            startangle=90,
            colors=["#16A085", "#E76F51"],
            wedgeprops={
                "width": 0.45,
                "edgecolor": "white"
            }
        )

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    left, right = st.columns(2)

    with left:
        st.subheader("Monthly Charges Distribution")

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.hist(
            df["MonthlyCharges"].dropna(),
            bins=20,
            color="#4776D0",
            edgecolor="white"
        )

        ax.set_xlabel("Monthly Charges")
        ax.set_ylabel("Customers")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with right:
        st.subheader("Customer Tenure")

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.hist(
            df["TenureMonths"].dropna(),
            bins=20,
            color="#8B72C7",
            edgecolor="white"
        )

        ax.set_xlabel("Tenure (Months)")
        ax.set_ylabel("Customers")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Recent Customer Records")

    st.dataframe(
        df.head(10),
        use_container_width=True,
        hide_index=True
    )


# ==================================================
# CUSTOMER DATA
# ==================================================
elif page == "Customer Data":

    st.subheader("Customer Explorer")
    st.caption("Search, filter, view and download customer records.")

    search = st.text_input("Search by Customer ID")

    col1, col2 = st.columns(2)

    with col1:
        status = st.selectbox(
            "Customer Status",
            ["All", "Active", "Churned"]
        )

    with col2:
        rows = st.selectbox(
            "Rows to display",
            [25, 50, 100, 500]
        )

    filtered = df.copy()

    if search.strip():
        filtered = filtered[
            filtered["CustomerID"].astype(str).str.contains(
                search.strip(),
                case=False,
                na=False
            )
        ]

    if status == "Active":
        filtered = filtered[filtered["Churn"] == 0]

    elif status == "Churned":
        filtered = filtered[filtered["Churn"] == 1]

    st.metric("Matching Customers", f"{len(filtered):,}")

    st.dataframe(
        filtered.head(rows),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇️ Download Customer Data CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="customer_records.csv",
        mime="text/csv"
    )


# ==================================================
# CHURN PREDICTION
# ==================================================
else:

    st.subheader("AI-Powered Churn Prediction")
    st.write("Enter customer information to estimate churn risk.")

    if not MODEL_FILE.exists():
        st.error("Model file not found: saas_churn_model.pkl")
        st.info(
            "Upload saas_churn_model.pkl to the same "
            "GitHub repository as app.py."
        )
        st.stop()

    try:
        model = load_model()
    except Exception as e:
        st.error(f"Unable to load the trained model: {e}")
        st.stop()

    # Keep feature names and order consistent with the existing model.
    feature_names = [
        "TenureMonths",
        "MonthlyCharges",
        "UsageHours",
        "SupportTickets",
        "PaymentDelays"
    ]

    # Check model feature names if available.
    trained_features = getattr(model, "feature_names_in_", None)

    if trained_features is not None:
        if list(trained_features) != feature_names:
            st.error(
                "The model's feature names or order do not match "
                "the prediction form. The model must use: "
                + ", ".join(feature_names)
            )
            st.stop()

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:
            tenure = st.number_input(
                "Tenure (Months)",
                min_value=0,
                max_value=120,
                value=12,
                step=1
            )

            charges = st.number_input(
                "Monthly Charges",
                min_value=0.0,
                value=1000.0,
                step=50.0
            )

            usage = st.number_input(
                "Usage Hours",
                min_value=0.0,
                value=30.0,
                step=1.0
            )

        with col2:
            tickets = st.number_input(
                "Support Tickets",
                min_value=0,
                value=2,
                step=1
            )

            delays = st.number_input(
                "Payment Delays",
                min_value=0,
                value=1,
                step=1
            )

        submitted = st.form_submit_button(
            "Predict Churn Risk",
            use_container_width=True
        )

    if submitted:

        try:
            customer = pd.DataFrame(
                [[tenure, charges, usage, tickets, delays]],
                columns=feature_names
            )

            prediction = int(model.predict(customer)[0])

            probability = None

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(customer)[0]
                classes = list(model.classes_)

                if 1 in classes:
                    churn_index = classes.index(1)
                    probability = float(
                        probabilities[churn_index]
                    ) * 100

            if probability is not None:
                st.metric(
                    "Estimated Churn Probability",
                    f"{probability:.1f}%"
                )

                st.progress(
                    min(max(probability / 100, 0.0), 1.0)
                )

            if prediction == 1:
                st.error("⚠️ Higher Risk: Customer may churn.")
                st.write(
                    "Review customer support issues, "
                    "usage patterns and payment delays."
                )

            elif prediction == 0:
                st.success(
                    "✅ Lower Risk: Customer is likely to stay."
                )

            else:
                st.warning(
                    f"The model returned an unexpected result: "
                    f"{prediction}"
                )

            st.caption(
                "This is an estimate from the trained model, "
                "not a guarantee of customer behavior."
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")


# ---------------- FOOTER ----------------
st.sidebar.divider()
st.sidebar.caption("SaaS Churn Analytics Demo")
