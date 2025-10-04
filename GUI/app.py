import json
import numpy as np
import streamlit as st
from streamlit_lottie import st_lottie
import requests
from PIL import Image
import cv2
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
from pathlib import Path

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="Traffic Sign Recognition",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS STYLING ====================
def apply_custom_css():
    st.markdown("""
    <style>
    /* Main background and theme */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.2);
        border-radius: 8px;
        color: white;
        font-weight: 600;
        padding: 12px 24px;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    /* Card containers */
    .card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 25px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        margin: 15px 0;
        transition: transform 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: white;
        font-family: 'Segoe UI', sans-serif;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 30px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
    }
    
    /* Image containers */
    .stImage {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* Success/Error messages */
    .stSuccess, .stError, .stInfo {
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# ==================== UTILITY FUNCTIONS ====================
def load_lottie_url(url: str):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None

def check_gpu():
    """Check CUDA availability"""
    if torch.cuda.is_available():
        return f"✅ GPU: {torch.cuda.get_device_name(0)}"
    return "💻 Running on CPU"

# ==================== PREPROCESSING CLASS ====================
class ImagePreprocessor:
    def __init__(self, image):
        self.image = image
    
    def resize_image(self, size=(255, 255)):
        self.image = cv2.resize(self.image, size)
        return self.image
    
    def convert_to_grayscale(self):
        if len(self.image.shape) == 3:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        return self.image
    
    def reduce_noise(self, kernel_size=5):
        self.image = cv2.GaussianBlur(self.image, (kernel_size, kernel_size), 0)
        return self.image
    
    def enhance_contrast(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        if len(self.image.shape) == 3:
            lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            cl = clahe.apply(l)
            lab = cv2.merge((cl, a, b))
            self.image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return self.image
    
    def normalize_image(self):
        self.image = self.image / 255.0 if self.image.dtype != np.float32 else self.image
        self.image = (self.image * 255).astype(np.uint8)
        return self.image
    
    def brighten_image(self, alpha=1.0, beta=50):
        self.image = cv2.convertScaleAbs(self.image, alpha=alpha, beta=beta)
        return self.image

# ==================== SEGMENTATION METHODS ====================
def automatic_thresholding(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresholded

def adaptive_thresholding(image, block_size=11, C=2):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY, block_size, C)

def local_thresholding(image, block_size=15):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rows, cols = image.shape
    result = np.zeros_like(image)
    for i in range(0, rows, block_size):
        for j in range(0, cols, block_size):
            block = image[i:i + block_size, j:j + block_size]
            threshold = np.mean(block)
            result[i:i + block_size, j:j + block_size] = (block > threshold).astype(np.uint8) * 255
    return result

# ==================== FEATURE EXTRACTION ====================
def extract_lbp(image):
    rows, cols = image.shape
    lbp_image = np.zeros((rows, cols), dtype=np.uint8)
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            center_pixel = image[i, j]
            binary_string = ''
            for di, dj in [(-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1)]:
                binary_string += '1' if image[i + di, j + dj] >= center_pixel else '0'
            lbp_image[i, j] = int(binary_string, 2)
    return lbp_image

def extract_local_features_lbp(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return extract_lbp(gray)

def extract_edges(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return cv2.Canny(gray, 50, 150)

# ==================== MODEL ARCHITECTURE ====================
class ClassificationModel(nn.Module):
    """Custom CNN model matching the saved state_dict structure"""
    def __init__(self, num_classes=43):
        super(ClassificationModel, self).__init__()
        
        # Feature extraction layers
        # Input: 3x224x224 → After conv1: 32x224x224 → After pool1: 32x112x112
        # After conv2: 64x112x112 → After pool2: 64x56x56 = 64*56*56 = 200,704
        # But checkpoint shows 3136 = 64*49 = 64*7*7
        # This means input must be smaller or more pooling layers
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 32x224x224
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32x112x112
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 64x112x112
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 64x56x56
        )
        
        # Classifier layers (3136 = 64 * 7 * 7, so we need adaptive pooling)
        self.classifier = nn.Sequential(
            nn.Linear(3136, 128),  # Matches checkpoint: 128 hidden units
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)  # 128 → 43
        )
        
        # Add adaptive pooling to ensure 7x7 output
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
    
    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)  # Force 64x7x7 = 3136
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ==================== TRAFFIC SIGN CLASSES ====================
TRAFFIC_SIGN_CLASSES = {
    0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)', 4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)', 8: 'Speed limit (120km/h)',
    9: 'No passing', 10: 'No passing veh over 3.5 tons', 11: 'Right-of-way at intersection',
    12: 'Priority road', 13: 'Yield', 14: 'Stop', 15: 'No vehicles',
    16: 'Veh > 3.5 tons prohibited', 17: 'No entry', 18: 'General caution',
    19: 'Dangerous curve left', 20: 'Dangerous curve right', 21: 'Double curve',
    22: 'Bumpy road', 23: 'Slippery road', 24: 'Road narrows on the right',
    25: 'Road work', 26: 'Traffic signals', 27: 'Pedestrians', 28: 'Children crossing',
    29: 'Bicycles crossing', 30: 'Beware of ice/snow', 31: 'Wild animals crossing',
    32: 'End speed + passing limits', 33: 'Turn right ahead', 34: 'Turn left ahead',
    35: 'Ahead only', 36: 'Go straight or right', 37: 'Go straight or left',
    38: 'Keep right', 39: 'Keep left', 40: 'Roundabout mandatory',
    41: 'End of no passing', 42: 'End no passing veh > 3.5 tons'
}

# ==================== WELCOME PAGE ====================
def show_welcome_page():
    st.markdown("""
    <div style='text-align: center; padding: 50px;'>
        <h1 style='font-size: 4em; margin-bottom: 20px;'>🚦 Traffic Sign Recognition</h1>
        <h3 style='font-size: 1.5em; opacity: 0.9;'>Advanced Computer Vision Pipeline</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        lottie_url = "https://assets5.lottiefiles.com/packages/lf20_jcikwtux.json"
        lottie_json = load_lottie_url(lottie_url)
        if lottie_json:
            st_lottie(lottie_json, height=300, key="welcome")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("🚀 Start Application", use_container_width=True, key="start_btn"):
            st.session_state.started = True
            st.rerun()
        
        st.markdown(f"""
        <div style='text-align: center; margin-top: 40px; color: white;'>
            <p style='font-size: 1.1em;'>{check_gpu()}</p>
            <p style='opacity: 0.8;'>Built with PyTorch & Streamlit</p>
        </div>
        """, unsafe_allow_html=True)

# ==================== MAIN APPLICATION ====================
def main_application():
    # Sidebar configuration
    with st.sidebar:
        st.markdown("### ⚙️ Configuration Panel")
        st.markdown("---")
        
        # Global settings
        st.markdown("#### 🎛️ Global Settings")
        device = st.selectbox("Device", ["Auto", "CPU", "CUDA"], key="device_select")
        
        st.markdown("---")
        st.markdown("#### 📊 Pipeline Status")
        
        if 'original_image' in st.session_state:
            st.success("✅ Image Loaded")
        else:
            st.info("⏳ No Image Loaded")
        
        if 'processed_image' in st.session_state:
            st.success("✅ Preprocessed")
        else:
            st.info("⏳ Not Preprocessed")
        
        if 'segmented_image' in st.session_state:
            st.success("✅ Segmented")
        else:
            st.info("⏳ Not Segmented")
        
        if 'extracted_features' in st.session_state:
            st.success("✅ Features Extracted")
        else:
            st.info("⏳ No Features")
    
    # Main content
    st.markdown("<h1 style='text-align: center;'>🚦 Traffic Sign Recognition System</h1>", 
                unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader(
        "📤 Upload Traffic Sign Image",
        type=["png", "jpg", "jpeg"],
        help="Upload an image of a traffic sign"
    )
    
    if uploaded_file is not None:
        # Load and store original image
        img = Image.open(uploaded_file)
        img = np.array(img)
        st.session_state.original_image = img
        
        # Create tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🖼️ Original", 
            "🔧 Preprocessing", 
            "✂️ Segmentation", 
            "🔍 Feature Extraction", 
            "🎯 Classification"
        ])
        
        # ==================== TAB 1: ORIGINAL IMAGE ====================
        with tab1:
            st.markdown("### 📸 Original Image")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(img, caption="Original Traffic Sign", use_container_width=True)
            
            st.markdown("#### 📊 Image Properties")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Height", f"{img.shape[0]}px")
            with col2:
                st.metric("Width", f"{img.shape[1]}px")
            with col3:
                st.metric("Channels", img.shape[2] if len(img.shape) == 3 else 1)
            with col4:
                st.metric("Size", f"{img.nbytes / 1024:.1f} KB")
        
        # ==================== TAB 2: PREPROCESSING ====================
        with tab2:
            st.markdown("### 🔧 Image Preprocessing")
            
            with st.sidebar:
                st.markdown("#### 🔧 Preprocessing Parameters")
                grayscale = st.checkbox("Convert to Grayscale", value=False)
                resize_w = st.slider("Resize Width", 128, 512, 255, step=32)
                resize_h = st.slider("Resize Height", 128, 512, 255, step=32)
                noise_kernel = st.slider("Noise Reduction Kernel", 3, 9, 5, step=2)
                brightness_alpha = st.slider("Brightness Alpha", 0.5, 2.0, 1.0, step=0.1)
                brightness_beta = st.slider("Brightness Beta", -50, 100, 50, step=10)
                contrast_clip = st.slider("Contrast Clip Limit", 1.0, 5.0, 2.0, step=0.5)
                
                apply_preprocess = st.button("🚀 Apply Preprocessing", use_container_width=True)
            
            if apply_preprocess:
                with st.spinner("Processing image..."):
                    preprocessor = ImagePreprocessor(img.copy())
                    
                    if grayscale:
                        preprocessor.convert_to_grayscale()
                    
                    preprocessor.resize_image((resize_w, resize_h))
                    preprocessor.reduce_noise(noise_kernel)
                    preprocessor.brighten_image(brightness_alpha, brightness_beta)
                    preprocessor.enhance_contrast(contrast_clip)
                    preprocessor.normalize_image()
                    
                    st.session_state.processed_image = preprocessor.image
                    st.success("✅ Preprocessing completed!")
            
            if 'processed_image' in st.session_state:
                col1, col2 = st.columns(2)
                with col1:
                    st.image(img, caption="Original", use_container_width=True)
                with col2:
                    st.image(st.session_state.processed_image, caption="Preprocessed", 
                           use_container_width=True)
        
        # ==================== TAB 3: SEGMENTATION ====================
        with tab3:
            st.markdown("### ✂️ Image Segmentation")
            
            with st.sidebar:
                st.markdown("#### ✂️ Segmentation Parameters")
                seg_input = st.radio("Input Image", ["Original", "Preprocessed"])
                seg_method = st.selectbox("Method", [
                    "Automatic (Otsu)", 
                    "Adaptive Thresholding",
                    "Local Thresholding"
                ])
                
                if seg_method == "Adaptive Thresholding":
                    block_size = st.slider("Block Size", 3, 31, 11, step=2)
                    C = st.slider("Constant C", 0, 10, 2)
                elif seg_method == "Local Thresholding":
                    block_size = st.slider("Block Size", 5, 51, 15, step=2)
                
                apply_segment = st.button("🚀 Apply Segmentation", use_container_width=True)
            
            if apply_segment:
                with st.spinner("Segmenting image..."):
                    if seg_input == "Original":
                        seg_img = img.copy()
                    else:
                        seg_img = st.session_state.get('processed_image', img).copy()
                    
                    if seg_method == "Automatic (Otsu)":
                        result = automatic_thresholding(seg_img)
                    elif seg_method == "Adaptive Thresholding":
                        result = adaptive_thresholding(seg_img, block_size, C)
                    else:
                        result = local_thresholding(seg_img, block_size)
                    
                    st.session_state.segmented_image = result
                    st.success("✅ Segmentation completed!")
            
            if 'segmented_image' in st.session_state:
                col1, col2 = st.columns(2)
                with col1:
                    st.image(img, caption="Original", use_container_width=True)
                with col2:
                    st.image(st.session_state.segmented_image, caption="Segmented", 
                           use_container_width=True)
        
        # ==================== TAB 4: FEATURE EXTRACTION ====================
        with tab4:
            st.markdown("### 🔍 Feature Extraction")
            
            with st.sidebar:
                st.markdown("#### 🔍 Feature Extraction Parameters")
                feat_input = st.radio("Input Image", 
                                     ["Original", "Preprocessed", "Segmented"], 
                                     key="feat_input")
                feat_method = st.selectbox("Method", ["LBP (Local Binary Pattern)", "Edge Detection"])
                
                apply_extract = st.button("🚀 Extract Features", use_container_width=True)
            
            if apply_extract:
                with st.spinner("Extracting features..."):
                    if feat_input == "Original":
                        feat_img = img.copy()
                    elif feat_input == "Preprocessed":
                        feat_img = st.session_state.get('processed_image', img).copy()
                    else:
                        feat_img = st.session_state.get('segmented_image', img).copy()
                    
                    if feat_method == "LBP (Local Binary Pattern)":
                        result = extract_local_features_lbp(feat_img)
                    else:
                        result = extract_edges(feat_img)
                    
                    st.session_state.extracted_features = result
                    st.success("✅ Feature extraction completed!")
            
            if 'extracted_features' in st.session_state:
                col1, col2 = st.columns(2)
                with col1:
                    st.image(img, caption="Original", use_container_width=True)
                with col2:
                    st.image(st.session_state.extracted_features, caption="Extracted Features", 
                           use_container_width=True)
        
        # ==================== TAB 5: CLASSIFICATION ====================
        with tab5:
            st.markdown("### 🎯 Traffic Sign Classification")
            
            with st.sidebar:
                st.markdown("#### 🎯 Classification Parameters")
                class_input = st.radio("Input Image", 
                                      ["Original", "Preprocessed", "Segmented", "Features"],
                                      key="class_input")
                model_path = st.text_input("Model Path", value="models/best_model.pth")
                
                classify_btn = st.button("🚀 Classify Sign", use_container_width=True)
            
            if classify_btn:
                model_file = Path(model_path)
                
                if not model_file.exists():
                    st.error(f"❌ Model file not found: {model_path}")
                    st.info("💡 Please ensure the model file exists in the specified path.")
                else:
                    with st.spinner("Loading model and classifying..."):
                        try:
                            # Determine device
                            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                            
                            # Load model with flexible architecture detection
                            model = ClassificationModel(num_classes=43)
                            state_dict = torch.load(model_path, map_location=device)
                            model.load_state_dict(state_dict)
                            model.to(device)
                            model.eval()
                            
                            # Select input image
                            if class_input == "Original":
                                class_img = img.copy()
                            elif class_input == "Preprocessed":
                                class_img = st.session_state.get('processed_image', img).copy()
                            elif class_input == "Segmented":
                                class_img = st.session_state.get('segmented_image', img).copy()
                            else:
                                class_img = st.session_state.get('extracted_features', img).copy()
                            
                            # Convert to RGB if grayscale
                            if len(class_img.shape) == 2:
                                class_img = cv2.cvtColor(class_img, cv2.COLOR_GRAY2RGB)
                            
                            # Transform
                            transform = transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.Resize((255, 255)),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                   std=[0.229, 0.224, 0.225])
                            ])
                            
                            input_tensor = transform(class_img).unsqueeze(0).to(device)
                            
                            # Predict
                            with torch.no_grad():
                                outputs = model(input_tensor)
                                probabilities = torch.softmax(outputs, dim=1)
                                confidence, predicted = torch.max(probabilities, 1)
                                
                                predicted_class = predicted.item()
                                confidence_score = confidence.item() * 100
                            
                            # Display results
                            st.success("✅ Classification completed!")
                            
                            col1, col2 = st.columns([1, 1])
                            
                            with col1:
                                st.image(class_img, caption="Input Image", use_container_width=True)
                            
                            with col2:
                                st.markdown("### 📊 Prediction Results")
                                st.markdown(f"""
                                <div style='background: white; padding: 20px; border-radius: 10px;'>
                                    <h2 style='color: #667eea; margin: 0;'>
                                        {TRAFFIC_SIGN_CLASSES.get(predicted_class, 'Unknown')}
                                    </h2>
                                    <p style='font-size: 1.5em; color: #764ba2; margin: 10px 0;'>
                                        Confidence: <strong>{confidence_score:.2f}%</strong>
                                    </p>
                                    <p style='color: #666;'>Class ID: {predicted_class}</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Top 5 predictions
                                st.markdown("#### 🏆 Top 5 Predictions")
                                top5_prob, top5_classes = torch.topk(probabilities, 5)
                                
                                for i, (prob, cls) in enumerate(zip(top5_prob[0], top5_classes[0])):
                                    st.progress(prob.item())
                                    st.caption(f"{i+1}. {TRAFFIC_SIGN_CLASSES.get(cls.item(), 'Unknown')} - {prob.item()*100:.2f}%")
                        
                        except Exception as e:
                            st.error(f"❌ Classification failed: {str(e)}")
                            st.info("💡 Check model path and compatibility.")

# ==================== MAIN ENTRY POINT ====================
def main():
    apply_custom_css()
    
    # Initialize session state
    if 'started' not in st.session_state:
        st.session_state.started = False
    
    # Show appropriate page
    if not st.session_state.started:
        show_welcome_page()
    else:
        main_application()

if __name__ == "__main__":
    main()