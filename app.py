import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 3rem;
    }
    
    .prediction-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .prediction-text {
        color: white;
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        margin: 0;
    }
    
    .probability-text {
        color: white;
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
    }
    
    .churn-high {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
    }
    
    .churn-low {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    
    .input-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #4ecdc4;
        margin-bottom: 1rem;
    }
    
    .section-header {
        font-size: 1.3rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model_and_encoders():
    model = tf.keras.models.load_model('model.h5')
    
    with open('label_encoder_gender.pkl', 'rb') as file:
        label_encoder_gender = pickle.load(file)
    
    with open('onehot_encoder_geo.pkl', 'rb') as file:
        onehot_encoder_geo = pickle.load(file)
    
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    
    return model, label_encoder_gender, onehot_encoder_geo, scaler

# Load models and encoders
model, label_encoder_gender, onehot_encoder_geo, scaler = load_model_and_encoders()

# Main header
st.markdown('<h1 class="main-header">🎯 Customer Churn Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Predict customer churn probability using ANN</p>', unsafe_allow_html=True)

# Create two columns for better layout
col1, col2 = st.columns([2, 1])

with col1:
    # Personal Information Section
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">👤 Personal Information</div>', unsafe_allow_html=True)
    
    col_personal1, col_personal2 = st.columns(2)
    
    with col_personal1:
        geography = st.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
        gender = st.selectbox('👥 Gender', label_encoder_gender.classes_)
        age = st.slider('🎂 Age', 18, 92, value=35)
    
    with col_personal2:
        tenure = st.slider('⏰ Tenure (years)', 0, 10, value=5)
        credit_score = st.number_input('💳 Credit Score', min_value=300, max_value=850, value=650)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Financial Information Section
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">💰 Financial Information</div>', unsafe_allow_html=True)
    
    col_financial1, col_financial2 = st.columns(2)
    
    with col_financial1:
        balance = st.number_input('💵 Account Balance', min_value=0.0, value=50000.0, format="%.2f")
        estimated_salary = st.number_input('💼 Estimated Salary', min_value=0.0, value=75000.0, format="%.2f")
    
    with col_financial2:
        num_of_products = st.slider('📦 Number of Products', 1, 4, value=2)
        has_cr_card = st.selectbox('💳 Has Credit Card', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
        is_active_member = st.selectbox('⭐ Is Active Member', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    # Prediction Section
    st.markdown("### 🔮 Prediction")
    
    # Create a predict button
    if st.button("🚀 Predict Churn", type="primary", use_container_width=True):
        # Prepare the input data
        input_data = pd.DataFrame({
            'CreditScore': [credit_score],
            'Gender': [label_encoder_gender.transform([gender])[0]],
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_of_products],
            'HasCrCard': [has_cr_card],
            'IsActiveMember': [is_active_member],
            'EstimatedSalary': [estimated_salary]
        })
        
        # One-hot encode 'Geography'
        geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
        geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
        
        # Combine one-hot encoded columns with input data
        input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
        
        # Scale the input data
        input_data_scaled = scaler.transform(input_data)
        
        # Predict churn
        prediction = model.predict(input_data_scaled)
        prediction_proba = prediction[0][0]
        
        # Display prediction with styling
        churn_class = "churn-high" if prediction_proba > 0.5 else "churn-low"
        
        st.markdown(f'''
        <div class="prediction-container {churn_class}">
            <p class="probability-text">{prediction_proba:.1%}</p>
            <p class="prediction-text">Churn Probability</p>
        </div>
        ''', unsafe_allow_html=True)
        
        # Additional insights
        if prediction_proba > 0.5:
            st.error('⚠️ High Risk: The customer is likely to churn!')
            st.markdown("**Recommended Actions:**")
            st.markdown("- Reach out with retention offers")
            st.markdown("- Provide personalized customer service")
            st.markdown("- Consider loyalty programs")
        else:
            st.success('✅ Low Risk: The customer is not likely to churn.')
            st.markdown("**Customer Status:**")
            st.markdown("- Maintain current service level")
            st.markdown("- Consider upselling opportunities")
            st.markdown("- Monitor for changes in behavior")
        
        # Risk gauge
        st.markdown("### 📊 Risk Level")
        
        if prediction_proba > 0.8:
            st.error("🔴 Very High Risk")
        elif prediction_proba > 0.6:
            st.warning("🟡 High Risk")
        elif prediction_proba > 0.4:
            st.info("🟢 Medium Risk")
        else:
            st.success("✅ Low Risk")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; padding: 1rem;'>"
    "Built with ❤️ using Streamlit & TensorFlow"
    "</div>", 
    unsafe_allow_html=True
)