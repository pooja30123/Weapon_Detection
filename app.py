import streamlit as st

# Set page config
st.set_page_config(
    page_title="Weapon Detection System",
    page_icon=" ",
    layout="wide"
)

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import os
import time



# Custom CSS for dark theme with security system aesthetic
st.markdown("""
<style>
    /* Page background - darker black theme */
    .main {
        background-color: #000000;
        font-family: 'Roboto Mono', 'Courier New', monospace;
    }
    
    /* Override Streamlit's default background */
    .stApp {
        background-color: #000000;
    }
    
    /* Ensure full page coverage with black */
    body {
        background-color: #000000;
        color: #CCCCCC;
    }
    
    /* Header styling */
    h1 {
        color: #FFFFFF;
        text-align: center;
        font-weight: 700;
        margin-bottom: 1.5rem;
        font-size: 1.75rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        border-bottom: 2px solid #FF3B30;
        padding-bottom: 0.75rem;
    }
    
    h2, h3, h4 {
        color: #D0D0D0;
        font-weight: 600;
        margin-top: 1.5rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    p, li {
        color: #A0A0A0;
        line-height: 1.5;
    }
    
    /* Security status indicators */
    .status-secure {
        color: #32CD32;
        font-weight: bold;
    }
    
    .status-warning {
        color: #FFC107;
        font-weight: bold;
    }
    
    .status-alert {
        color: #FF3B30;
        font-weight: bold;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #111111;
        color: white;
        border: 1px solid #FF3B30;
        padding: 0.5rem 1rem;
        border-radius: 3px;
        font-weight: 500;
        font-family: 'Roboto Mono', monospace;
        transition: all 0.2s ease;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .stButton>button:hover {
        background-color: #FF3B30;
        box-shadow: 0 0 8px rgba(255, 59, 48, 0.6);
    }
    
    /* File uploader styling */
    .stFileUploader>div {
        border-radius: 3px;
        border: 1px dashed #444444;
        padding: 1.5rem;
        text-align: center;
        background-color: #111111;
    }
    
    .stFileUploader>div:hover {
        border-color: #FF3B30;
    }
    
    .stFileUploader>div>div>button {
        background-color: #111111;
        color: white;
        border: 1px solid #FF3B30;
        padding: 0.5rem 1rem;
        border-radius: 3px;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .stFileUploader>div>div>button:hover {
        background-color: #FF3B30;
    }
    
    /* Input widgets styling */
    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        background-color: #111111;
        color: #D0D0D0;
        border: 1px solid #333333;
        border-radius: 3px;
        padding: 0.5rem 0.75rem;
        font-family: 'Roboto Mono', monospace;
    }
    
    .stTextInput>div>div>input:focus, .stNumberInput>div>div>input:focus {
        border-color: #FF3B30;
        box-shadow: 0 0 0 2px rgba(255, 59, 48, 0.2);
    }
    
    /* Select box styling */
    .stSelectbox>div>div>div {
        background-color: #111111;
        color: #D0D0D0;
        border: 1px solid #333333;
        border-radius: 3px;
        font-family: 'Roboto Mono', monospace;
    }
    
    /* Sidebar styling */
    .css-1d391kg, .css-12oz5g7 {
        background-color: #080808;
    }
    
    .css-1d391kg .sidebar-content, .css-12oz5g7 .sidebar-content {
        background-color: #080808;
    }
    
    /* Card/container elements */
    div.stDataFrame, div[data-testid="stBlock"] {
        border-radius: 3px;
        background-color: #0D0D0D;
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid #222222;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.5);
    }
    
    /* Security camera/detection feed styling */
    .detection-frame {
        border: 2px solid #FF3B30;
        position: relative;
        padding: 10px;
        background-color: #080808;
    }
    
    .detection-frame::before {
        content: "LIVE";
        position: absolute;
        top: 5px;
        right: 10px;
        color: #FF3B30;
        font-size: 12px;
        font-weight: bold;
        animation: blink 1s infinite;
    }
    
    @keyframes blink {
        0% { opacity: 1; }
        50% { opacity: 0; }
        100% { opacity: 1; }
    }
    
    /* Security grid overlay for images */
    .security-grid {
        background-image: linear-gradient(rgba(255, 59, 48, 0.1) 1px, transparent 1px),
                          linear-gradient(90deg, rgba(255, 59, 48, 0.1) 1px, transparent 1px);
        background-size: 20px 20px;
        position: relative;
    }
    
    /* Timestamp styling */
    .timestamp {
        font-family: 'Roboto Mono', monospace;
        color: #FF3B30;
        font-size: 12px;
        margin-top: 5px;
    }
    
    /* Metrics and data visualization */
    div[data-testid="stMetric"] {
        background-color: #0D0D0D;
        border-left: 3px solid #FF3B30;
        padding: 10px;
    }
    
    div[data-testid="stMetric"] label {
        color: #777777 !important;
        text-transform: uppercase;
        font-size: 12px;
        letter-spacing: 0.05em;
    }
    
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-family: 'Roboto Mono', monospace;
    }
    
    /* Additional elements styling for black theme */
    .stCheckbox, .stRadio {
        background-color: #0D0D0D;
        padding: 10px;
        border-radius: 3px;
        border: 1px solid #222222;
    }
    
    .stDateInput>div>div>input {
        background-color: #111111;
        border: 1px solid #333333;
        color: #D0D0D0;
    }
    
    /* Navigation and footer styling */
    .stApp footer {
        background-color: #000000;
        color: #555555;
        border-top: 1px solid #222222;
    }
    
    /* Table styling */
    .stTable table {
        background-color: #0D0D0D;
        color: #D0D0D0;
        border: 1px solid #222222;
    }
    
    .stTable th {
        background-color: #111111;
        color: #FFFFFF;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-size: 0.9em;
    }
    
    .stTable td {
        border-color: #222222;
    }
    
    /* Weapon detection highlight */
    .weapon-detected {
        border: 2px solid #FF3B30;
        background-color: rgba(255, 59, 48, 0.1);
        padding: 5px;
        animation: alert-pulse 2s infinite;
    }
    
    @keyframes alert-pulse {
        0% { box-shadow: 0 0 0 0 rgba(255, 59, 48, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(255, 59, 48, 0); }
        100% { box-shadow: 0 0 0 0 rgba(255, 59, 48, 0); }
    }
    
    /* Horizontal lines styling */
    hr {
        border: 0;
        height: 1px;
        background-image: linear-gradient(to right, rgba(255, 59, 48, 0), rgba(255, 59, 48, 0.75), rgba(255, 59, 48, 0));
        margin: 1.5rem 0;
    }
    
    /* Container for detection display */
    .detection-container {
        border: 1px solid #333333;
        background-color: #0D0D0D;
        padding: 10px;
        border-radius: 3px;
    }
    
    /* Custom header with icon */
    .icon-header {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Title with security system styling
st.markdown("""
<div class="icon-header">
    <h1>🔫 WEAPON DETECTION SYSTEM</h1>
</div>
<div class="timestamp" style="text-align: center;">SYSTEM ONLINE • SECURITY LEVEL: ACTIVE</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Sidebar for system information
with st.sidebar:
    st.markdown("<h3>SYSTEM STATUS</h3>", unsafe_allow_html=True)
    st.markdown("<div class='status-secure'>● SYSTEM ONLINE</div>", unsafe_allow_html=True)
    st.markdown("<div class='timestamp'>Last updated: {}</div>".format(time.strftime("%H:%M:%S")), unsafe_allow_html=True)
    
    st.markdown("<h3>DETECTION SETTINGS</h3>", unsafe_allow_html=True)
    confidence_threshold = st.slider("Detection Confidence", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
    show_bbox = st.checkbox("Show Bounding Boxes", value=True)
    enable_alerts = st.checkbox("Enable Alert Sounds", value=False)
    
    st.markdown("<h3>SYSTEM INFO</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size: 0.85rem; color: #888888;">
    MODEL: YOLOv8 Custom<br>
    VERSION: v2.0<br>
    CLASSES: Weapons<br>
    </div>
    """, unsafe_allow_html=True)

# Load YOLO model
@st.cache_resource
def load_model():
    model = YOLO(r'C:\Users\POOJA\Desktop\Weapon-Detection\model\version3.pt')  # Load YOLOv8 model
    return model

try:
    model = load_model()
    st.sidebar.markdown("<div class='status-secure'>● MODEL LOADED</div>", unsafe_allow_html=True)
except Exception as e:
    st.sidebar.markdown("<div class='status-alert'>● MODEL ERROR</div>", unsafe_allow_html=True)
    st.sidebar.error(f"Error loading model: {e}")

def detect_weapons(frame, conf_threshold=0.5):
    # Convert frame to proper format based on input type
    if isinstance(frame, Image.Image):
        # Convert PIL Image to numpy array in RGB format
        frame_np = np.array(frame)
        # Convert RGB to BGR for OpenCV processing
        frame = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
    
    # Get frame dimensions for timestamp
    height, width = frame.shape[:2]
    
    # Add timestamp to frame
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, timestamp, (10, height - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 59, 48), 1)
    
    # Security grid overlay (optional)
    # Add grid lines to frame
    grid_size = 30
    for i in range(0, width, grid_size):
        cv2.line(frame, (i, 0), (i, height), (255, 59, 48, 20), 1)
    for j in range(0, height, grid_size):
        cv2.line(frame, (0, j), (width, j), (255, 59, 48, 20), 1)
    
    # Perform detection with YOLO model
    results = model(frame)
    
    # Flag to check if any weapons are detected
    weapons_detected = False
    
    # Draw bounding boxes
    for detection in results[0].boxes:
        x1, y1, x2, y2 = map(int, detection.xyxy[0].tolist())
        conf = detection.conf[0].item()
        cls = int(detection.cls[0].item())
        
        if conf > conf_threshold:  # Confidence threshold
            weapons_detected = True
            if show_bbox:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # Add more prominent alert label
                cv2.putText(frame, f"WEAPON DETECTED", (x1, y1 - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, f"CONF: {conf:.2f}", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Add warning label if weapons detected
    if weapons_detected:
        warning_text = "WARNING: WEAPON DETECTED"
        text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = (width - text_size[0]) // 2
        
        # Draw warning banner
        cv2.rectangle(frame, (0, 0), (width, 40), (0, 0, 255), -1)
        cv2.putText(frame, warning_text, (text_x, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
    return frame, weapons_detected

# ====================== IMAGE DETECTION ======================
st.markdown("""
<h2>📷 IMAGE DETECTION</h2>
<div class="timestamp">Upload an image to scan for weapons</div>
""", unsafe_allow_html=True)

image_col1, image_col2 = st.columns(2)

with image_col1:
    uploaded_file = st.file_uploader("Upload Security Image", type=["jpg", "png", "jpeg"], key="image_uploader")

if uploaded_file is not None:
    # Load the image
    image = Image.open(uploaded_file)
    
    with image_col1:
        st.markdown("<div class='detection-container'>", unsafe_allow_html=True)
        st.image(image, caption="Original Image", use_column_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    if st.button("SCAN FOR WEAPONS"):
        with st.spinner("Analyzing image..."):
            # Process the image for weapon detection
            processed_img, weapons_found = detect_weapons(image, conf_threshold=confidence_threshold)
            
            # Convert BGR back to RGB for displaying in Streamlit
            processed_img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
            
        with image_col2:
            st.markdown("<div class='detection-container'>", unsafe_allow_html=True)
            # Display processed image with RGB color order
            st.image(processed_img_rgb, caption="Detection Results", use_column_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            if weapons_found:
                st.markdown("<div class='status-alert'>⚠️ WEAPON DETECTED IN IMAGE</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='status-secure'>✓ NO WEAPONS DETECTED</div>", unsafe_allow_html=True)
            
# ====================== VIDEO DETECTION ======================
st.markdown("---")
st.markdown("""
<h2>🎥 VIDEO DETECTION</h2>
<div class="timestamp">Analyze video footage for weapons</div>
""", unsafe_allow_html=True)

# Video options
video_option = st.radio("Select Video Source", ["Upload Video", "Use Webcam"])

if video_option == "Upload Video":
    video_file = st.file_uploader("Upload Security Footage", type=["mp4", "mov", "avi"], key="video_uploader")
    
    if video_file is not None:
        st.markdown("<div class='detection-frame'>", unsafe_allow_html=True)
        video_placeholder = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)
        
        status_placeholder = st.empty()
        
        process_button = st.button("START VIDEO ANALYSIS")
        stop_button = st.button("STOP ANALYSIS")
        
        if process_button:
            status_placeholder.markdown("<div class='status-warning'>⚙️ ANALYZING VIDEO...</div>", unsafe_allow_html=True)
            
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            
            cap = cv2.VideoCapture(tfile.name)
            frame_count = 0
            weapon_frames = 0
            
            while cap.isOpened() and not stop_button:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                processed_frame, weapons_detected = detect_weapons(frame, conf_threshold=confidence_threshold)
                
                if weapons_detected:
                    weapon_frames += 1
                    status_placeholder.markdown("<div class='status-alert'>⚠️ WEAPON DETECTED</div>", unsafe_allow_html=True)
                else:
                    status_placeholder.markdown("<div class='status-secure'>✓ SCANNING</div>", unsafe_allow_html=True)
                
                # Convert BGR to RGB for display
                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(rgb_frame, use_column_width=True)
                time.sleep(0.03)  # Slow down video playback
            
            cap.release()
            os.unlink(tfile.name)
            
            if frame_count > 0:
                weapon_percentage = (weapon_frames / frame_count) * 100
                st.markdown(f"""
                <div style="background-color: #111111; padding: 15px; border-radius: 3px; margin-top: 20px;">
                    <h4>ANALYSIS RESULTS</h4>
                    <p>Frames analyzed: {frame_count}</p>
                    <p>Frames with weapons: {weapon_frames} ({weapon_percentage:.1f}%)</p>
                </div>
                """, unsafe_allow_html=True)

elif video_option == "Use Webcam":
    st.markdown("<div class='status-warning'>⚠️ Webcam access may require browser permissions</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='detection-frame'>", unsafe_allow_html=True)
    webcam_placeholder = st.empty()
    st.markdown("</div>", unsafe_allow_html=True)
    
    status_placeholder = st.empty()
    
    start_webcam = st.button("START WEBCAM MONITORING")
    stop_webcam = st.button("STOP MONITORING")
    
    if start_webcam:
        status_placeholder.markdown("<div class='status-warning'>⚙️ INITIALIZING WEBCAM...</div>", unsafe_allow_html=True)
        
        try:
            cam = cv2.VideoCapture(0)
            
            if not cam.isOpened():
                status_placeholder.markdown("<div class='status-alert'>❌ FAILED TO ACCESS WEBCAM</div>", unsafe_allow_html=True)
            else:
                status_placeholder.markdown("<div class='status-secure'>✓ WEBCAM MONITORING ACTIVE</div>", unsafe_allow_html=True)
                
                while cam.isOpened() and not stop_webcam:
                    ret, frame = cam.read()
                    if not ret:
                        status_placeholder.markdown("<div class='status-alert'>❌ WEBCAM DISCONNECTED</div>", unsafe_allow_html=True)
                        break
                    
                    processed_frame, weapons_detected = detect_weapons(frame, conf_threshold=confidence_threshold)
                    
                    if weapons_detected:
                        status_placeholder.markdown("<div class='status-alert'>⚠️ WEAPON DETECTED</div>", unsafe_allow_html=True)
                    else:
                        status_placeholder.markdown("<div class='status-secure'>✓ MONITORING</div>", unsafe_allow_html=True)
                    
                    # Convert BGR to RGB for display
                    rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    webcam_placeholder.image(rgb_frame, use_column_width=True)
                
                cam.release()
        except Exception as e:
            status_placeholder.markdown(f"<div class='status-alert'>❌ ERROR: {e}</div>", unsafe_allow_html=True)

# System footer
st.markdown("---")
st.markdown("""
<div style="display: flex; justify-content: space-between; color: #555555; font-size: 0.8rem;">
    <div>WEAPON DETECTION SYSTEM v2.0</div>
    <div>© 2025 SECURITY SYSTEMS</div>
</div>
""", unsafe_allow_html=True)