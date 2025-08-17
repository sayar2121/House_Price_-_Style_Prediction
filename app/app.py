# app/app.py
import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(
    page_title="HomeValue AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for a Professional Look ---
st.markdown("""
<style>
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }
    /* Sidebar */
    .st-emotion-cache-16txtl3 {
        background-color: #F0F2F6;
    }
    /* Metric cards */
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: scale(1.05);
    }
    .metric-title {
        font-size: 1.2em;
        font-weight: bold;
        color: #333333;
    }
    .metric-value {
        font-size: 2.5em;
        font-weight: bold;
        color: #007BFF;
    }
    .metric-value-style {
        font-size: 2.5em;
        font-weight: bold;
        color: #28A745;
    }
    /* Header image */
    .header-image {
        width: 100%;
        height: 250px;
        object-fit: cover;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Load Models ---
@st.cache_resource
def load_model(model_name):
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, model_name)
        return joblib.load(model_path)
    except FileNotFoundError:
        st.error(f"❌ Error: Model '{model_name}' not found.")
        st.warning("Please run `train_all_models.py` first.", icon="🏃")
        return None

price_model = load_model("house_price_model.pkl")
style_model = load_model("house_style_model.pkl")

if not price_model or not style_model:
    st.stop()

# --- Sidebar Inputs ---
with st.sidebar:
    st.title("🤖 HomeValue AI")
    st.write("Adjust the features below to get a prediction.")
    st.markdown("---")

    lot_area = st.number_input("Lot Area (sq ft)", 500, 200000, 8450, 100)
    overall_qual = st.slider("Overall Quality", 1, 10, 7)
    year_built = st.number_input("Year Built", 1872, 2024, 2003)
    gr_liv_area = st.number_input("Living Area (sq ft)", 334, 5642, 1710, 50)
    full_bath = st.slider("Full Bathrooms", 0, 4, 2)
    bedroom_abvgr = st.slider("Bedrooms", 0, 8, 3)
    garage_cars = st.slider("Garage Capacity (cars)", 0, 4, 2)
    
    predict_button = st.button("🔮 Predict House Value", type="primary", use_container_width=True)

# --- Main Page ---

# Header Image
st.markdown(
    f'<img src="https://images.unsplash.com/photo-1580587771525-78b9dba3b914?q=80&w=1974&auto=format&fit=crop" class="header-image" alt="Modern house banner">',
    unsafe_allow_html=True
)

st.title("Advanced House Insights Dashboard")
st.markdown("This dashboard uses AI to predict house prices and styles. Adjust the features on the left and click predict.")

# --- Prediction Display ---
if predict_button:
    input_data = pd.DataFrame({
        "LotArea": [lot_area], "OverallQual": [overall_qual],
        "YearBuilt": [year_built], "GrLivArea": [gr_liv_area],
        "FullBath": [full_bath], "BedroomAbvGr": [bedroom_abvgr],
        "GarageCars": [garage_cars]
    })

    price_prediction = price_model.predict(input_data)[0]
    style_prediction = style_model.predict(input_data)[0]

    st.markdown("---")
    st.header("Prediction Results")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Estimated Price</div>
            <div class="metric-value">${price_prediction:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Predicted House Style</div>
            <div class="metric-value-style">{style_prediction}</div>
        </div>
        """, unsafe_allow_html=True)

    # --- Feature Importance ---
    st.markdown("---")
    st.header("What Influenced the Price?")
    st.write("This chart shows which features had the biggest impact on the price prediction. Higher values mean more influence.")

    importances = price_model.feature_importances_
    feature_names = input_data.columns
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)

    fig = px.bar(
        feature_importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Feature Importance for Price Prediction',
        labels={'Importance': 'Relative Importance', 'Feature': 'House Feature'},
        template='plotly_white'
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Please adjust the features in the sidebar and click the **Predict House Value** button to see the results.", icon="👈")

# --- Footer ---
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:grey;'>Powered by 🤖 HomeValue AI</p>", unsafe_allow_html=True)
