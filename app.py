import streamlit as st
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from ultralytics import YOLO
import tempfile
import os

# --------------------------------------------
# Load YOLOv8 model
# --------------------------------------------
@st.cache_resource
def load_yolo_model():
    try:
        model = YOLO('./yolov8n.pt')
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None

model = load_yolo_model()

# --------------------------------------------
# Helper Functions
# --------------------------------------------
def load_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR), image

def compute_similarity(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    score, diff = ssim(gray1, gray2, full=True)
    return score, diff

def compute_color_hist_diff(img1, img2):
    """Compute color histogram difference (0-1, higher means more different)."""
    hist1 = cv2.calcHist([img1], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
    hist2 = cv2.calcHist([img2], [0,1,2], None, [8,8,8], [0,256,0,256,0,256])
    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()
    corr = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return 1 - corr  # 0-1, higher diff

def compute_edge_diff(img1, img2):
    """Compute edge difference score."""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    edges1 = cv2.Canny(gray1, 100, 200)
    edges2 = cv2.Canny(gray2, 100, 200)
    diff = cv2.absdiff(edges1, edges2)
    return np.sum(diff) / (diff.shape[0] * diff.shape[1])  # average diff

def detect_objects(img):
    """Detect objects using YOLO."""
    if model is None:
        return []
    results = model(img)
    if len(results) > 0 and len(results[0].boxes) > 0:
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        return [model.names[cls_id] for cls_id in class_ids]
    return []

def compare_objects(objects1, objects2):
    """Compare object lists."""
    set1 = set(objects1)
    set2 = set(objects2)
    missing = set1 - set2
    added = set2 - set1
    return missing, added

def highlight_differences_both(img1, img2, diff, objects1, objects2, missing, added, color_diff, edge_diff):
    """Highlight differences on both images with boxes and labels."""
    highlighted1 = img1.copy()
    highlighted2 = img2.copy()

    # Draw YOLO boxes: red for img1, green for img2
    yolo_boxes1 = []
    yolo_boxes2 = []
    if model:
        results1 = model(img1)
        results2 = model(img2)
        for r in results1:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cv2.rectangle(highlighted1, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)  # red
                yolo_boxes1.append((int(x1), int(y1), int(x2), int(y2)))
        for r in results2:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cv2.rectangle(highlighted2, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)  # green
                yolo_boxes2.append((int(x1), int(y1), int(x2), int(y2)))

    # Always add SSIM diff contours for all changes, with higher sensitivity
    diff_uint8 = (diff * 255).astype("uint8")
    thresh = cv2.threshold(diff_uint8, 80, 255, cv2.THRESH_BINARY_INV)[1]  # fixed lower threshold for small changes like bindi
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) > 50:  # much lower area to catch small changes like bindi
            x, y, w, h = cv2.boundingRect(c)
            # Check if contour overlaps with YOLO box (person, etc.)
            overlap1 = any(x < bx2 and x+w > bx1 and y < by2 and y+h > by1 for bx1, by1, bx2, by2 in yolo_boxes1)
            overlap2 = any(x < bx2 and x+w > bx1 and y < by2 and y+h > by1 for bx1, by1, bx2, by2 in yolo_boxes2)
            if overlap1 or overlap2:
                color = (255, 255, 0)  # yellow for object-related diffs (e.g., costume changes like bindi)
            else:
                color = (255, 0, 0)  # blue for background/property diffs
            cv2.rectangle(highlighted1, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(highlighted2, (x, y), (x + w, y + h), color, 2)

    return highlighted1, highlighted2

def extract_keyframes(video_path, interval=5):
    """Extract frames every 'interval' seconds."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval)
    frames = []
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_interval == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    return frames

def compare_videos(video1_path, video2_path, interval=30):
    frames1 = extract_keyframes(video1_path, interval)
    frames2 = extract_keyframes(video2_path, interval)
    min_len = min(len(frames1), len(frames2))

    similarities = []
    diff_frames = []
    reports = []

    for i in range(min_len):
        f1 = cv2.resize(frames1[i], (640, 360))
        f2 = cv2.resize(frames2[i], (640, 360))
        score, diff = compute_similarity(f1, f2)
        color_diff = compute_color_hist_diff(f1, f2)
        edge_diff = compute_edge_diff(f1, f2)
        similarities.append(score)

        # Detect objects
        objects1 = detect_objects(f1)
        objects2 = detect_objects(f2)
        missing, added = compare_objects(objects1, objects2)

        # Determine if mismatch - lower thresholds for more sensitivity
        is_mismatch = score < 0.95 or color_diff > 0.2 or edge_diff > 0.05 or (missing or added)

        if is_mismatch:
            highlighted1, highlighted2 = highlight_differences_both(f1, f2, diff, objects1, objects2, missing, added, color_diff, edge_diff)
            diff_frames.append((i, highlighted1, highlighted2))

            # Generate detailed report
            report = f"Frame {i}: "
            if missing:
                report += f"Objects missing: {list(missing)}. "
            if added:
                report += f"Objects added: {list(added)}. "
            if color_diff > 0.2:
                report += f"Lighting/color change (diff: {color_diff:.2f}). "
            if edge_diff > 0.05:
                report += f"Structural/edge change (diff: {edge_diff:.2f}). "
            if score < 0.95:
                report += f"Overall similarity low ({score:.2f}). "
            reports.append(report.strip())

    return similarities, diff_frames, reports

# --------------------------------------------
# Streamlit UI
# --------------------------------------------
st.title("🎬 ContinuAIty Checker")
st.write("Compare film frames or full scenes (videos) for visual continuity — background, lighting, and set consistency across days of shooting.")

uploaded1 = st.file_uploader("📁 Upload First Image or Video", type=['png', 'jpg', 'jpeg', 'mp4', 'mov', 'avi'])
uploaded2 = st.file_uploader("📁 Upload Second Image or Video", type=['png', 'jpg', 'jpeg', 'mp4', 'mov', 'avi'])

if st.button("🔍 Analyze Continuity") and uploaded1 and uploaded2:
    temp1 = tempfile.NamedTemporaryFile(delete=False)
    temp2 = tempfile.NamedTemporaryFile(delete=False)
    temp1.write(uploaded1.read())
    temp2.write(uploaded2.read())
    temp1.close()
    temp2.close()

    is_video = uploaded1.type.startswith("video") or uploaded2.type.startswith("video")

    if is_video:
        st.subheader("🎥 Scene Comparison Preview")
        col1, col2 = st.columns(2)
        with col1:
            st.video(temp1.name)
        with col2:
            st.video(temp2.name)

        st.info("⏳ Analyzing scenes... please wait (processing keyframes every 30 seconds for speed)...")
        similarities, diff_frames, reports = compare_videos(temp1.name, temp2.name, interval=30)
        avg_sim = np.mean(similarities) * 100
        st.write(f"**🎞️ Average Scene Similarity:** {avg_sim:.2f}%")

        if diff_frames:
            st.warning("⚠️ Scene mismatches detected:")
            for i, (h1, h2) in diff_frames:
                col1, col2 = st.columns(2)
                with col1:
                    st.image(cv2.cvtColor(h1, cv2.COLOR_BGR2RGB),
                             caption=f"Frame {i} — Video 1 (Red: objects, Green: added objects, Blue: background diffs, Yellow: object-related diffs)",
                             use_container_width=True)
                with col2:
                    st.image(cv2.cvtColor(h2, cv2.COLOR_BGR2RGB),
                             caption=f"Frame {i} — Video 2 (Red: objects, Green: added objects, Blue: background diffs, Yellow: object-related diffs)",
                             use_container_width=True)
                # Display report for this frame
                if reports:
                    st.write(f"**Report for Frame {i}:** {reports[i]}")
        else:
            st.success("✅ No major mismatches found — scene continuity looks good!")

        # Overall report
        if reports:
            st.subheader("📋 Continuity Report Summary")
            for rep in reports:
                st.write(f"- {rep}")

        os.remove(temp1.name)
        os.remove(temp2.name)

    else:
        # ------------------ IMAGE MODE ------------------
        img1_cv, img1_pil = load_image(uploaded1)
        img2_cv, img2_pil = load_image(uploaded2)
        height, width = img1_cv.shape[:2]
        img2_cv = cv2.resize(img2_cv, (width, height))

        col1, col2 = st.columns(2)
        with col1:
            st.image(img1_pil, caption="📸 First Image", use_container_width=True)
        with col2:
            st.image(img2_pil, caption="📸 Second Image", use_container_width=True)

        score, diff = compute_similarity(img1_cv, img2_cv)
        color_diff = compute_color_hist_diff(img1_cv, img2_cv)
        edge_diff = compute_edge_diff(img1_cv, img2_cv)
        st.write(f"**🧮 Image Similarity:** {score * 100:.2f}%")

        # Detect objects
        objects1 = detect_objects(img1_cv)
        objects2 = detect_objects(img2_cv)
        missing, added = compare_objects(objects1, objects2)

        # Highlight on both
        highlighted1, highlighted2 = highlight_differences_both(img1_cv, img2_cv, diff, objects1, objects2, missing, added, color_diff, edge_diff)

        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(highlighted1, cv2.COLOR_BGR2RGB),
                     caption="📸 First Image (Red: objects, Green: added objects, Blue: background diffs, Yellow: object-related diffs)", use_container_width=True)
        with col2:
            st.image(cv2.cvtColor(highlighted2, cv2.COLOR_BGR2RGB),
                     caption="📸 Second Image (Red: objects, Green: added objects, Blue: background diffs, Yellow: object-related diffs)", use_container_width=True)

        # Report
        report = ""
        if missing:
            report += f"Objects missing: {list(missing)}. "
        if added:
            report += f"Objects added: {list(added)}. "
        if color_diff > 0.2:
            report += f"Lighting/color change (diff: {color_diff:.2f}). "
        if edge_diff > 0.05:
            report += f"Structural/edge change (diff: {edge_diff:.2f}). "
        if score < 0.95:
            report += f"Overall similarity low ({score:.2f}). "
        if report:
            st.write(f"**📋 Image Report:** {report.strip()}")
        else:
            st.success("✅ No significant differences detected.")

elif st.button("🔄 Reset"):
     st.rerun()

