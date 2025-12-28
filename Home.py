import streamlit as st
from PIL import Image

st.set_page_config(
    page_title = "Facial Analysis System - Emotion Recognition and Beauty Score Prediction",
    page_icon = "🎭",
    layout = "wide",
    initial_sidebar_state = "expanded"
)

st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(100deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    .info-card {
        background-color: #1e1e1e;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    h3 {
        font-size: 28px !important;
        font-weight: 700;
    }

    h4 {
        font-size: 22px !important;
        font-weight: 600;
    }

    /* Paragraph text */
    p, li {
        font-size: 18px !important;
        line-height: 1.6;
    }

    /* Info cards */
    .info-card {
        font-size: 18px;
        padding: 16px;
    }

    /* Make bullet points more readable */
    ul {
        padding-left: 22px;
    }
    </style>
""", unsafe_allow_html=True)

# Main Header
st.markdown("""
    <div class="main-header">
        <h2 style="color: white; font-size: 2.5rem;">🎭 Facial Analysis System: A system for beauty score prediction and emotion recognition.</h2>
    </div>
""", unsafe_allow_html=True)

 # Introduction Section
st.markdown("""
### Welcome to the Facial Analysis System!

This system allows you to upload facial images and receive instant predictions from both models, enabling direct comparison of their performance and insights.
    
**🎯 Primary Goals:**
- **Beauty Score Prediction**: Quantify facial attractiveness on a scale of 1.0 to 5.0
- **Emotion Recognition**: Classify facial expressions into 7 distinct emotions

**🔬 Research Focus:**
This project compares the performance of a single-task model for beauty score prediction with a multi-task model that jointly performs beauty score prediction and emotion classification.
    
""")

# How to Use Section
st.markdown("""
    <div>
        <h3 style="color: white; margin: 0;">📝 How to Use This System</h3>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="info-card">
        <h4 style="color: white;">Follow these simple steps to analyze facial images with our dual-model system:</h4>
        <ul> 
            <li><strong>Navigate to Analysis Page: </strong>Click on 'Model Analysis' in the sidebar to access the analysis interface</li>
            <li><strong>Upload Your Image: </strong>Choose a clear frontal face image (JPG, JPEG, or PNG format)</li>
            <li><strong>Wait for Processing: </strong>Both models will analyze the image simultaneously</li>
            <li><strong>Review Results: </strong>Compare beauty scores and view emotion predictions with confidence levels</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Best Practices
st.markdown("""
<div>
    <h3 style="color: white; margin: 0;">✨ Tips for Best Results</h3>
</div>
""", unsafe_allow_html=True)

tip_col1, tip_col2 = st.columns(2)

with tip_col1:
    st.markdown("""
    <div class="info-card">
        <h4 style="color: white;">📸 Image Quality</h4>
        <ul> 
            <li>Ensure good lighting conditions</li>
            <li>Avoid heavy shadows or overexposure</li>
            <li>Clear focus on the face</li>
        </ul>
    </div>
    <div class="info-card">
        <h4 style="color: white;">👤 Face Position</h4>
        <ul> 
            <li>Frontal face view works best</li>
            <li>Face should occupy at least 60% of image</li>
            <li>Minimal head tilt or rotation</li>
            <li>Eyes should be clearly visible</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with tip_col2:
    st.markdown("""
    <div class="info-card">
        <h4 style="color: white;">🎨 Image Characteristics</h4>
        <ul> 
            <li>Neutral background preferred</li>
            <li>Avoid heavy filters or editing</li>
            <li>Remove obstructions (sunglasses, masks)</li>
        </ul>
    </div>
    <div class="info-card">
        <h4 style="color: white;">⚠️ What to Avoid</h4>
        <ul> 
            <li>Group photos (multiple faces)</li>
            <li>Side profile or extreme angles</li>
            <li>Low resolution or blurry images</li>
            <li>Cartoon or illustrated faces</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray; padding: 1rem;'>
        <p style="font-size: 0.9rem;">Built with Streamlit | Powered by PyTorch</p>
        <p style="font-size: 0.85rem;">Comparing Deep Learning Single-Task vs Multi-Task Learning for Facial Analysis</p>
    </div>
""", unsafe_allow_html=True)