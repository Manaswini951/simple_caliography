import streamlit as st
import cv2
import numpy as np
import tempfile
import imageio.v2 as imageio

st.set_page_config(page_title="Calligraphy Stroke Animator", layout="centered")
st.title("🖋️ Calligraphy Stroke Animator")
st.write("Upload your calligraphy photo to extract the strokes, animate them being drawn on a clean white canvas, and reveal enhanced colors.")

uploaded_file = st.file_uploader("Choose a calligraphy image...", type=["jpg", "jpeg", "png"])

def clean_and_extract_strokes(bgr_img):
    h, w = bgr_img.shape[:2]
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

    # Background illumination correction
    kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    bg = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel_bg)
    diff = cv2.absdiff(bg, gray)
    norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Otsu thresholding on normalized ink
    _, stroke_mask = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Clean up small noise and edge border artifacts
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    stroke_mask = cv2.morphologyEx(stroke_mask, cv2.MORPH_OPEN, kernel_clean)

    border = 15
    stroke_mask[:border, :] = 0
    stroke_mask[-border:, :] = 0
    stroke_mask[:, :border] = 0
    stroke_mask[:, -border:] = 0

    # Filter connected components by minimum area
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(stroke_mask, connectivity=8)
    min_stroke_area = (h * w) * 0.001
    clean_mask = np.zeros_like(stroke_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_stroke_area:
            clean_mask[labels == i] = 255

    # Boost color vibrancy & lighting
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.45, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.15, 0, 255)
    enhanced_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return clean_mask, enhanced_bgr

def build_animation(bgr_img, stroke_mask, enhanced_bgr, total_draw_frames=45, glow_frames=15):
    h, w = stroke_mask.shape
    coords = np.argwhere(stroke_mask > 0)
    if len(coords) == 0:
        return []

    diag_scores = coords[:, 0] * 0.6 + coords[:, 1] * 0.4
    sorted_order = np.argsort(diag_scores)
    sorted_coords = coords[sorted_order]

    frames = []

    # Phase 1: Progressive drawing on white canvas
    chunk_size = int(np.ceil(len(sorted_coords) / total_draw_frames))
    canvas_mask = np.zeros((h, w), dtype=np.uint8)

    for f in range(total_draw_frames):
        idx_end = min((f + 1) * chunk_size, len(sorted_coords))
        new_pts = sorted_coords[f * chunk_size:idx_end]
        if len(new_pts) > 0:
            canvas_mask[new_pts[:, 0], new_pts[:, 1]] = 255

        soft_mask = cv2.GaussianBlur(canvas_mask, (5, 5), 0) / 255.0
        frame_rgb = np.zeros((h, w, 3), dtype=np.float32)

        for c in range(3):
            stroke_ch = enhanced_bgr[:, :, 2 - c].astype(np.float32)
            frame_rgb[:, :, c] = 255.0 * (1.0 - soft_mask) + stroke_ch * soft_mask

        frames.append(np.clip(frame_rgb, 0, 255).astype(np.uint8))

    # Phase 2: Color glow & brightening
    final_draw = frames[-1].copy()
    soft_full_mask = cv2.GaussianBlur(stroke_mask, (7, 7), 0) / 255.0
    glow_mask = cv2.GaussianBlur(stroke_mask, (25, 25), 0) / 255.0

    for f in range(glow_frames):
        alpha = (f + 1) / glow_frames
        blended = final_draw.astype(np.float32)

        for c in range(3):
            orig_ch = enhanced_bgr[:, :, 2 - c].astype(np.float32)
            bloom = cv2.GaussianBlur(orig_ch, (21, 21), 0)
            glow_layer = orig_ch * (1.0 - 0.25 * alpha) + bloom * (0.25 * alpha)

            blended[:, :, c] = (
                255.0 * (1.0 - soft_full_mask)
                + (glow_layer * soft_full_mask) * (1.0 + 0.15 * alpha * glow_mask)
            )

        frames.append(np.clip(blended, 0, 255).astype(np.uint8))

    for _ in range(10):
        frames.append(frames[-1])

    return frames

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Downscale and ensure even dimensions for MP4 encoding
    max_dim = 900
    h, w = image_bgr.shape[:2]
    scale = min(1.0, max_dim / float(max(h, w)))
    new_w = int(w * scale) // 2 * 2
    new_h = int(h * scale) // 2 * 2
    image_bgr = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    with st.spinner("Extracting calligraphy strokes and rendering video..."):
        mask, enhanced_bgr = clean_and_extract_strokes(image_bgr)
        frames = build_animation(image_bgr, mask, enhanced_bgr)

    if frames:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_file:
            imageio.mimsave(tmp_file.name, frames, fps=24)
            st.success("Animation complete!")
            st.video(tmp_file.name)
    else:
        st.error("No calligraphy strokes detected. Try adjusting lighting or contrast.")
