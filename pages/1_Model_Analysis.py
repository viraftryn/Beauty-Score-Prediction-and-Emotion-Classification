import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

class BeautyRegressionModel(nn.Module):
    """EfficientNet-B0 based Beauty Score Regression Model"""
    def __init__(self):
        super().__init__()
        
        # Load EfficientNet-B0 base
        try:
            from torchvision.models import EfficientNet_B0_Weights
            base_model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        except:
            base_model = models.efficientnet_b0(pretrained=True)
        
        # Extract components directly (no wrapper)
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        
        num_features = base_model.classifier[1].in_features  # 1280
        
        # Replace classifier with regression head (exactly as in training)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(p=0.2),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        # EfficientNet forward pass
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x.squeeze(1)

class EfficientNetMTLModel(nn.Module):
    """EfficientNet-B0 Multi-Task Learning Model"""
    def __init__(self, num_emotions=7):
        super().__init__()
        
        self.backbone = models.efficientnet_b0(pretrained=False)
        in_features = 1280
        self.backbone.classifier = nn.Identity()
        
        self.shared_neck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.regression_branch = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
        
        self.classification_branch = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_emotions)
        )

    def forward(self, x):
        features = self.backbone(x)
        shared = self.shared_neck(features)
        beauty_score = self.regression_branch(shared).squeeze(1)
        emotion_logits = self.classification_branch(shared)
        return emotion_logits, beauty_score

    def predict(self, x):
        emotion_logits, beauty_score = self.forward(x)
        emotion_probs = torch.softmax(emotion_logits, dim=1)
        return emotion_probs, beauty_score

# ============================================================================
# CONSTANTS
# ============================================================================

EMOTION_LABELS = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

EMOTION_COLORS = {
    "Angry": "#FF6B6B",
    "Disgust": "#95E1D3",
    "Fear": "#A8E6CF",
    "Happy": "#FFD93D",
    "Sad": "#6C5CE7",
    "Surprise": "#FDA7DF",
    "Neutral": "#95A5A6"
}

# ============================================================================
# MODEL LOADING
# ============================================================================

@st.cache_resource
def load_models():
    """Load both models"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    models_dict = {
        'beauty': None,
        'mtl': None
    }
    
    # Load Beauty Regression Model
    try:
        beauty_model = BeautyRegressionModel()
        checkpoint = torch.load('best_efficientnet_scut_model.pth', 
                          map_location=device, weights_only=False)
    
        # Load the model state dict properly
        if 'model_state_dict' in checkpoint:
            beauty_model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        else:
            beauty_model.load_state_dict(checkpoint, strict=True)
    
        beauty_model.to(device)
        beauty_model.eval()
        models_dict['beauty'] = beauty_model
    except FileNotFoundError:
        st.warning("⚠️ Beauty regression model file not found: 'best_efficientnet_scut_model.pth'")
    except Exception as e:
        st.error(f"❌ Error loading beauty model: {str(e)[:200]}")
    
    # Load Multi-Task Learning Model
    try:
        mtl_model = EfficientNetMTLModel(num_emotions=7)
        checkpoint = torch.load('mtl_efficientnet_best.pth', 
                              map_location=device, weights_only=False)
        mtl_model.load_state_dict(checkpoint['model_state_dict'])
        mtl_model.to(device)
        mtl_model.eval()
        models_dict['mtl'] = mtl_model
    except FileNotFoundError:
        st.warning("⚠️ Multi-task model file not found: 'mtl_efficientnet_best.pth'")
    except Exception as e:
        st.error(f"❌ Error loading MTL model: {str(e)[:200]}")
    
    return models_dict, device

# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================

def preprocess_image_efficientnet_single(image, target_size=(224, 224)):
    """Preprocess image for EfficientNet-B0 single-task model - EXACT MATCH to training"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize
    image = image.resize(target_size)
    
    # Convert to array and normalize to [0, 1]
    img_array = np.array(image).astype(np.float32) / 255.0
    
    # Convert to tensor (C, H, W)
    img_tensor = torch.tensor(img_array).permute(2, 0, 1)
    
    # Apply ImageNet normalization (same as training)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225])
    img_tensor = normalize(img_tensor)
    
    # Add batch dimension
    return img_tensor.unsqueeze(0)

def preprocess_image_efficientnet_mtl(image, target_size=(48, 48)):
    """Preprocess image for EfficientNet-B0 (simple normalization)"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = image.resize(target_size)
    img_array = np.array(image).astype(np.float32) / 255.0
    img_tensor = torch.tensor(img_array).permute(2, 0, 1).unsqueeze(0)
    
    return img_tensor

# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def predict_beauty_only(model, image, device):
    """Predict beauty score using single-task EfficientNet model"""
    img_tensor = preprocess_image_efficientnet_single(image)  
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        beauty_score = model(img_tensor).cpu().item()
    
    return beauty_score

def predict_mtl(model, image, device):
    """Predict both emotion and beauty using MTL model"""
    img_tensor = preprocess_image_efficientnet_mtl(image)  
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        emotion_probs, beauty_score = model.predict(img_tensor)
    
    emotion_probs = emotion_probs.cpu().numpy()[0]
    beauty_score = beauty_score.cpu().item()
    
    # Denormalize beauty score (0-1 -> 1-5)
    beauty_score_denorm = beauty_score * 4.0 + 1.0
    
    return emotion_probs, beauty_score_denorm

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_beauty_comparison_chart(beauty_score, mtl_beauty_score):
    """Create comparison chart for beauty scores"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=['Single-Task', 'Multi-Task Learning'],
        y=[beauty_score, mtl_beauty_score],
        marker_color=['#3498DB', '#E74C3C'],
        text=[f'{beauty_score:.2f}', f'{mtl_beauty_score:.2f}'],
        textposition='outside',
        textfont=dict(size=16, color='white')
    ))
    
    fig.update_layout(
        title="Beauty Score Comparison",
        yaxis_title="Score (1.0 - 5.0)",
        yaxis_range=[0, 5.5],
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_emotion_chart(emotion_probs):
    """Create emotion probability chart"""
    emotions = [EMOTION_LABELS[i] for i in range(7)]
    colors = [EMOTION_COLORS[em] for em in emotions]
    
    top_emotion_idx = np.argmax(emotion_probs)
    
    fig = go.Figure(data=[
        go.Bar(
            x=emotions,
            y=emotion_probs * 100,
            marker_color=[colors[i] if i == top_emotion_idx else '#95A5A6' 
                         for i in range(7)],
            text=[f'{p*100:.1f}%' for p in emotion_probs],
            textposition='outside',
            textfont=dict(size=12, color='white')
        )
    ])
    
    fig.update_layout(
        title="Emotion Classification Probabilities",
        xaxis_title="Emotion",
        yaxis_title="Probability (%)",
        yaxis_range=[0, 110],
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# ============================================================================
# STREAMLIT UI
# ============================================================================

st.set_page_config(
    page_title="Model Analysis - Facial Analysis System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #1e1e1e;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #333;
        margin: 0.5rem 0;
    }
    .stProgress > div > div > div > div {
        background-color: #667eea;
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

    /* Make bullet points more readable */
    ul {
        padding-left: 22px;
    }

    section[data-testid="stSidebar"] {
        font-size: 14px;
    }

    /* Sidebar headers */
    section[data-testid="stSidebar"] h1 {
        font-size: 18px !important;
    }

    section[data-testid="stSidebar"] h2 {
        font-size: 16px !important;
    }

    section[data-testid="stSidebar"] h3 {
        font-size: 15px !important;
    }

    /* Sidebar markdown text */
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] li,
    section[data-testid="stSidebar"] span {
        font-size: 14px !important;
        line-height: 1.4;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div class="main-header">
        <h1>🔍 Model Analysis Interface</h1>
        <p style="color: white; font-size: 1.1rem;">Upload and Analyze Facial Images</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📊 Model Information")
    
    st.markdown("### Model 1: Single-Task Learning")
    st.markdown("""
    - **Architecture**: EfficientNet-B0 
    - **Task**: Beauty Score Prediction
    - **Dataset**: SCUT-FBP5500
    - **Output**: Score (1.0 - 5.0)
    """)
    
    st.divider()
    
    st.markdown("### Model 2: Multi-Task Learning")
    st.markdown("""
    - **Architecture**: EfficientNet-B0
    - **Tasks**: Emotion + Beauty
    - **Datasets**: FER-2013 + SCUT-FBP5500
    - **Outputs**: 7 Emotions + Beauty Score
    """)

# Main Content
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📸 Upload Image")
    
    uploaded_file = st.file_uploader(
        "Choose a face image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear frontal face image for best results"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Image info
        st.markdown("**Image Details:**")
        st.text(f"Size: {image.size[0]}×{image.size[1]}")
        st.text(f"Mode: {image.mode}")
        st.text(f"Format: {uploaded_file.type}")

with col2:
    st.subheader("Analysis Results")
    
    if uploaded_file is not None:
        # Load models
        models_dict, device = load_models()
        
        if models_dict['beauty'] is None and models_dict['mtl'] is None:
            st.error("❌ No models could be loaded. Please check that model files exist.")
        else:
            with st.spinner("🔄 Analyzing image with both models..."):
                results = {}
                
                # Predict with Beauty Regression Model
                if models_dict['beauty'] is not None:
                    try:
                        beauty_score = predict_beauty_only(
                            models_dict['beauty'], image, device
                        )
                        results['beauty'] = beauty_score
                    except Exception as e:
                        st.error(f"Error in beauty model: {str(e)}")
                
                # Predict with Multi-Task Model
                if models_dict['mtl'] is not None:
                    try:
                        emotion_probs, mtl_beauty = predict_mtl(
                            models_dict['mtl'], image, device
                        )
                        results['mtl_beauty'] = mtl_beauty
                        results['emotions'] = emotion_probs
                    except Exception as e:
                        st.error(f"Error in MTL model: {str(e)}")
            
            # Display Results
            if results:
                # Beauty Score Comparison
                if 'beauty' in results and 'mtl_beauty' in results:
                    st.markdown("### Beauty Score Comparison")
                    
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        col_label, col_help = st.columns([4, 1])
                        with col_label:
                            st.markdown("**Single-Task Model**")
                        with col_help:
                            st.markdown("""
                                <style>
                                .help-icon {
                                cursor: help;
                                display: inline-block;
                                }
                                </style>
                            """, unsafe_allow_html=True)
                            with st.popover("ℹ️"):
                                st.markdown("""
                                    **Single-Task Learning**
                                    This model focuses **only** on predicting beauty scores.
                                    It was trained exclusively on facial attractiveness ratings,
                                    learning features specifically optimized for beauty assessment.
                                """)
                        st.metric(
                            "",  
                            f"{results['beauty']:.2f}/5.0"
                        )
                        st.progress(results['mtl_beauty'] / 5.0)
                    
                    with col_b:
                        col_label, col_help = st.columns([4, 1])
                        with col_label:
                            st.markdown("**Multi-Task Model**")
                        with col_help:
                            st.markdown("""
                                <style>
                                .help-icon {
                                cursor: help;
                                display: inline-block;
                                }
                                </style>
                            """, unsafe_allow_html=True)
                            with st.popover("ℹ️"):
                                st.markdown("""
                                    **Multi-Task Learning**
                                    This model learns **two tasks simultaneously**: predicting beauty scores 
                                    AND recognizing emotions. 
                                    By learning both tasks together, the model gains additional insights from facial expressions, 
                                    which can help understand attractiveness better.
                                """)
                        st.metric(
                            "",
                            f"{results['mtl_beauty']:.2f}/5.0"
                        )
                        st.progress(results['mtl_beauty'] / 5.0)
                    
                    with col_c:
                        diff = abs(results['beauty'] - results['mtl_beauty'])
                        percentage = ((results['mtl_beauty'] - results['beauty']) / results['beauty'] * 100)
                        st.metric(
                            "Difference",
                            f"{diff:.2f}",
                            # delta=f"{percentage:.1f}%"
                        )
                    
                    # Comparison Chart
                    fig = create_beauty_comparison_chart(
                        results['beauty'], 
                        results['mtl_beauty']
                    )

                    fig.update_layout(
                        title_font_size=24,
                        font=dict(
                            size=18  
                        ),
                        xaxis=dict(
                            title_font_size=20,
                            tickfont=dict(size=16)
                        ),
                        yaxis=dict(
                            title_font_size=20,
                            tickfont=dict(size=16)
                        ),
                        legend=dict(font=dict(size=16))
                    )

                    st.plotly_chart(fig, use_container_width=True)

                
                elif 'beauty' in results:
                    st.markdown("### ⭐ Beauty Score")
                    st.metric("EfficientNet-B0", f"{results['beauty']:.2f}/5.0")
                    st.progress(results['beauty'] / 5.0)
                
                elif 'mtl_beauty' in results:
                    st.markdown("### ⭐ Beauty Score")
                    st.metric("EfficientNet-B0", f"{results['mtl_beauty']:.2f}/5.0")
                    st.progress(results['mtl_beauty'] / 5.0)
                
                # Emotion Results (MTL only)
                if 'emotions' in results:
                    st.markdown("---")
                    st.markdown("### 🎭 Emotion Recognition (MTL Model Only)")
                    
                    top_emotion_idx = np.argmax(results['emotions'])
                    top_emotion = EMOTION_LABELS[top_emotion_idx]
                    top_confidence = results['emotions'][top_emotion_idx] * 100
                    
                    col_e1, col_e2 = st.columns(2)
                    
                    with col_e1:
                        st.markdown("#### Detected Emotion")
                        st.markdown(f"## {top_emotion}")
                        st.markdown(f"**Confidence:** {top_confidence:.1f}%")
                    
                    with col_e2:
                        st.markdown("#### Top 3 Emotions")
                        top_3_idx = np.argsort(results['emotions'])[-3:][::-1]
                        for idx in top_3_idx:
                            emotion = EMOTION_LABELS[idx]
                            prob = results['emotions'][idx] * 100
                            st.markdown(f"**{emotion}**: {prob:.1f}%")
                    
                    # Emotion Chart
                    fig_emotion = create_emotion_chart(results['emotions'])
                    fig_emotion.update_layout(
                        title_font_size=24,
                        font=dict(
                            size=18  
                        ),
                        xaxis=dict(
                            title_font_size=20,
                            tickfont=dict(size=16)
                        ),
                        yaxis=dict(
                            title_font_size=20,
                            tickfont=dict(size=16)
                        ),
                        legend=dict(font=dict(size=16))
                    )
                    st.plotly_chart(fig_emotion, use_container_width=True)
    else:
        st.info("👆 Upload an image to start the analysis")
        
        st.markdown("### 💡 Tips for Best Results")
        st.markdown("""
        - Use clear, well-lit frontal face images
        - Ensure the face is the main subject
        - Avoid heavy filters or distortions
        - JPG, JPEG, and PNG formats supported
        """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray; padding: 1rem;'>
        <p style="font-size: 0.9rem;">Built with Streamlit | Powered by PyTorch</p>
        <p style="font-size: 0.85rem;">Comparing Deep Learning Single-Task vs Multi-Task Learning for Facial Analysis</p>
    </div>
""", unsafe_allow_html=True)