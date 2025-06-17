import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4a7c59;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #3a6a4a;
        color: white;
    }
    .stSelectbox, .stSlider {
        margin-bottom: 1.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 20px;
        border-radius: 8px 8px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4a7c59;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Load model and encoder
@st.cache_resource
def load_model():
    model = joblib.load('xgb_modelo_iris.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    return model, label_encoder

model, label_encoder = load_model()

# Species information
SPECIES_INFO = {
    "Iris-setosa": {
        "description": "Small flowers with wide sepals and short petals. Common in cold climates.",
        "habitat": "Northern America and Asia",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg"
    },
    "Iris-versicolor": {
        "description": "Medium-sized flowers with vibrant colors. Prefer moist soils.",
        "habitat": "Eastern North America",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg"
    },
    "Iris-virginica": {
        "description": "Large flowers with wavy petals. Tolerate swampy soils.",
        "habitat": "Southeastern United States",
        "image_url": "https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg"
    }
}

# Prediction function
def predict_species(input_data):
    try:
        # Ensure correct input shape
        input_array = np.array(input_data).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(input_array)
        species = label_encoder.inverse_transform(prediction)[0]
        
        # Get probabilities if available
        probs = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_array)[0]
            
        return species, probs, None
        
    except Exception as e:
        return None, None, str(e)

# Main app
st.title("🌺 Iris Flower Classification System")
st.markdown("""
This application uses XGBoost to classify Iris flowers into three species based on their measurements.
""")

# Create tabs
tab1, tab2, tab3 = st.tabs(["🔍 Manual Prediction", "📁 Batch Prediction", "ℹ️ Species Information"])

with tab1:
    st.header("Manual Measurement Input")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Sepal Characteristics")
        sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.1, 0.1)
        sepal_width = st.slider("Sepal width (cm)", 2.0, 5.0, 3.5, 0.1)
    
    with col2:
        st.subheader("Petal Characteristics")
        petal_length = st.slider("Petal length (cm)", 1.0, 4.0, 1.4, 0.1)
        petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 0.2, 0.1)
    
    if st.button("Classify Flower", type="primary", use_container_width=True):
        with st.spinner("Analyzing measurements..."):
            input_data = [sepal_length, sepal_width, petal_length, petal_width]
            species, probs, error = predict_species(input_data)
            
            if error:
                st.error(f"Prediction error: {error}")
            else:
                st.success(f"**Predicted Species:** {species}")
                
                # Display species info
                col_img, col_desc = st.columns([1, 2])
                with col_img:
                    st.image(SPECIES_INFO[species]["image_url"], 
                            caption=species, 
                            width=250)
                
                with col_desc:
                    st.markdown(f"""
                    **Description:** {SPECIES_INFO[species]["description"]}
                    
                    **Natural Habitat:** {SPECIES_INFO[species]["habitat"]}
                    """)
                
                # Show probabilities if available
                if probs is not None:
                    st.subheader("Classification Probabilities")
                    
                    prob_df = pd.DataFrame({
                        "Species": label_encoder.classes_,
                        "Probability": probs
                    }).sort_values("Probability", ascending=False)
                    
                    # Display table and chart side by side
                    col_table, col_chart = st.columns(2)
                    
                    with col_table:
                        st.dataframe(
                            prob_df.style.format({"Probability": "{:.2%}"}),
                            hide_index=True,
                            use_container_width=True
                        )
                    
                    with col_chart:
                        fig, ax = plt.subplots()
                        ax.barh(prob_df["Species"], prob_df["Probability"], color="#4a7c59")
                        ax.set_xlim(0, 1)
                        ax.set_xlabel("Probability")
                        ax.set_facecolor("#f8f9fa")
                        fig.patch.set_facecolor("#f8f9fa")
                        st.pyplot(fig)

with tab2:
    st.header("Batch Prediction from CSV")
    st.markdown("""
    Upload a CSV file containing measurements for multiple flowers. The file should include columns named:
    - `sepal_length`
    - `sepal_width`
    - `petal_length`
    - `petal_width`
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Check required columns
            required_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
            if not all(col in df.columns for col in required_cols):
                st.error(f"CSV must contain these columns: {', '.join(required_cols)}")
            else:
                st.success("File successfully uploaded!")
                st.dataframe(df.head(), use_container_width=True)
                
                if st.button("Predict All Samples", type="primary", use_container_width=True):
                    with st.spinner("Processing predictions..."):
                        # Reorder columns to match training data
                        df = df[required_cols]
                        
                        # Make predictions
                        predictions = model.predict(df.values)
                        df['Predicted_Species'] = label_encoder.inverse_transform(predictions)
                        
                        # Show results
                        st.subheader("Prediction Results")
                        st.dataframe(df, use_container_width=True)
                        
                        # Download button
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download Predictions",
                            data=csv,
                            file_name="iris_predictions.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

with tab3:
    st.header("Iris Species Information")
    
    selected_species = st.selectbox(
        "Select a species to learn more:",
        list(SPECIES_INFO.keys()),
        index=0
    )
    
    col_img, col_desc = st.columns([1, 2])
    
    with col_img:
        st.image(
            SPECIES_INFO[selected_species]["image_url"],
            caption=selected_species,
            width=200  # Reduced from 300 to 200
        )
    
    with col_desc:
        st.markdown(f"""
        ### {selected_species}
        
        **Description:**  
        {SPECIES_INFO[selected_species]["description"]}
        
        **Natural Habitat:**  
        {SPECIES_INFO[selected_species]["habitat"]}
        """)