import streamlit as st
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from ultralytics import YOLO
import tempfile
import os
import time
from concurrent.futures import ThreadPoolExecutor
import torch
from sklearn.cluster import KMeans
import time

# --- PAGE CONFIG ---
st.set_page_config(page_title="ContinuAIty Pro", layout="wide", page_icon="🎬")

# --- CUSTOM CSS FOR PROFESSIONAL LOOK ---
# --- CUSTOM CSS FOR PROFESSIONAL LOOK ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True) # <--- MAKE SURE THIS WORD IS CORRECT


# --- MODEL LOADING ---
@st.cache_resource
def load_yolo_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO('yolov8n.pt') 
    return model, device

model, device = load_yolo_model()

# --- NEW FEATURE: EXPLAINABILITY HEATMAP (For Lexsi Labs) ---
def generate_xai_heatmap(img2, diff):
    # Convert SSIM difference to a visual heatmap
    diff_uint8 = (diff * 255).astype("uint8")
    heatmap = cv2.applyColorMap(255 - diff_uint8, cv2.COLORMAP_JET)
    # Superimpose heatmap onto the image
    overlay = cv2.addWeighted(img2, 0.7, heatmap, 0.3, 0)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

# --- NEW FEATURE: COLOR PALETTE EXTRACTION (For BROCHILL) ---
def get_color_palette(img, clusters=5):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_flat = img_rgb.reshape(-1, 3)
    kmeans = KMeans(n_clusters=clusters, n_init=10)
    kmeans.fit(img_flat)
    return kmeans.cluster_centers_.astype(int)

# --- CORE ANALYTICS FUNCTIONS ---
def compute_metrics(img1, img2):
    # Ensure same size
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    score, diff = ssim(gray1, gray2, full=True)
    
    # YOLO Detection
    res1 = model(img1, verbose=False)[0]
    res2 = model(img2, verbose=False)[0]
    
    obj1 = [model.names[int(c)] for c in res1.boxes.cls]
    obj2 = [model.names[int(c)] for c in res2.boxes.cls]
    
    return score, diff, obj1, obj2, res1.plot(), res2.plot()

# --- UI LAYOUT ---
st.title("🎬 ContinuAIty Pro: AI-Driven Visual Auditor")
st.write("Advanced Multimodal Analysis for Film Production & Industrial Safety.")

with st.sidebar:
    st.header("⚙️ Analysis Settings")
    sensitivity = st.slider("SSIM Sensitivity Threshold", 0.80, 0.99, 0.95)
    st.info("Higher sensitivity catches smaller continuity breaks.")

col_u1, col_u2 = st.columns(2)
with col_u1:
    up1 = st.file_uploader("Upload Baseline (Shot A)", type=['jpg','png','mp4'])
with col_u2:
    up2 = st.file_uploader("Upload Comparison (Shot B)", type=['jpg','png','mp4'])

if up1 and up2:
    if st.button("🚀 Run Deep Continuity Audit"):
        t1 = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(up1.name)[1])
        t2 = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(up2.name)[1])
        t1.write(up1.read()); t2.write(up2.read())
        t1.close(); t2.close()
        
        def get_frame(path):
            # If it's a video, grab the first frame
            if path.lower().endswith(('.mp4', '.mov', '.avi')):
                cap = cv2.VideoCapture(path)
                ret, frame = cap.read()
                cap.release()
                return frame if ret else None
            # If it's an image, read normally
            return cv2.imread(path)

        img1 = get_frame(t1.name)
        img2 = get_frame(t2.name)

        # --- SAFETY CHECK ---
        if img1 is None or img2 is None:
            st.error("❌ Error: Could not read one of the files. Please ensure you are uploading valid images or videos.")
        else:
            with st.spinner("Analyzing pixels, objects, and color narratives..."):
                # Ensure they are the same size before processing
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
                
                score, diff, obj1, obj2, plot1, plot2 = compute_metrics(img1, img2)
                palette1 = get_color_palette(img1)
                palette2 = get_color_palette(img2)

            # --- RESULTS TABS ---
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Summary", "🔍 XAI Heatmap", "🤖 Object Logic", "🎨 Color Story"])

            with tab1:
                st.metric("Overall Similarity Score", f"{score*100:.2f}%")
                if score < sensitivity:
                    st.error("⚠️ Continuity Alert: Visual mismatch detected.")
                else:
                    st.success("✅ Continuity Verified.")

            with tab2:
                st.subheader("Mechanistic Interpretability: Error Heatmap")
                heatmap_img = generate_xai_heatmap(img2, diff)
                st.image(heatmap_img, use_container_width=True)

            with tab3:
                c1, c2 = st.columns(2)
                c1.image(cv2.cvtColor(plot1, cv2.COLOR_BGR2RGB), caption="Objects in Shot A")
                c2.image(cv2.cvtColor(plot2, cv2.COLOR_BGR2RGB), caption="Objects in Shot B")
                
                missing = set(obj1) - set(obj2)
                added = set(obj2) - set(obj1)
                if missing or added:
                    st.warning(f"Object Mismatch: Missing {missing} | Added {added}")

            with tab4:
                st.subheader("Color Narrative Consistency")
                def show_palette(palette):
                    pal_img = np.zeros((60, 300, 3), dtype=np.uint8)
                    step = 300 // len(palette)
                    for i, color in enumerate(palette):
                        pal_img[:, i*step:(i+1)*step] = color
                    return pal_img

                st.write("Shot A Palette")
                st.image(show_palette(palette1))
                st.write("Shot B Palette")
                st.image(show_palette(palette2))

        # Cleanup
        os.remove(t1.name); os.remove(t2.name)
        
        
if up1 and up2:
    if st.button("🚀 Start Live Synchronized Audit"):
        t1 = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(up1.name)[1])
        t2 = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(up2.name)[1])
        t1.write(up1.read()); t2.write(up2.read())
        t1.close(); t2.close()

        # Initialize Video Captures
        cap1 = cv2.VideoCapture(t1.name)
        cap2 = cv2.VideoCapture(t2.name)

        # UI Layout: Create columns for side-by-side view
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Shot A (Baseline)")
            frame_place1 = st.empty()  # Placeholder for Video 1
        with col2:
            st.subheader("Shot B (Comparison)")
            frame_place2 = st.empty()  # Placeholder for Video 2

        # Progress and Metrics below videos
        st.divider()
        m_col1, m_col2 = st.columns(2)
        score_place = m_col1.empty()
        alert_place = m_col2.empty()

        # --- SYNC LOOP ---
        try:
            while cap1.isOpened() and cap2.isOpened():
                ret1, img1 = cap1.read()
                ret2, img2 = cap2.read()

                if not ret1 or not ret2:
                    break

                # 1. Processing Logic
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
                
                # Run your existing YOLO and SSIM logic
                # For live view, we only analyze every 5th frame to keep it fast
                score, diff, obj1, obj2, plot1, plot2 = compute_metrics(img1, img2)

                # 2. Visualization
                # On the comparison side (col2), let's show the Heatmap or YOLO plot
                # To make it "cool", let's overlay the heatmap on Shot B
                xai_view = generate_xai_heatmap(img2, diff)

                # 3. Update the Live Display
                frame_place1.image(cv2.cvtColor(plot1, cv2.COLOR_BGR2RGB), use_container_width=True)
                frame_place2.image(xai_view, use_container_width=True)

                # 4. Update Metrics
                score_place.metric("Live Similarity Score", f"{score*100:.1f}%")
                
                if score < sensitivity:
                    alert_place.error(f"🚨 ALERT: Continuity Break detected!")
                else:
                    alert_place.success("✅ Frames are Consistent")

                # Small sleep to control playback speed (e.g., 24 FPS)
                # time.sleep(0.01) 

        finally:
            cap1.release()
            cap2.release()
            os.remove(t1.name); os.remove(t2.name)
            st.info("Audit Complete.")