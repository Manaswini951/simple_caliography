import io
import math
import zipfile
from collections import deque
import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Smooth Calligraphy Flow & Merge Animator",
    page_icon="✒️",
    layout="wide",
)

st.title("✒️ Smooth Calligraphy Flow & Merge Animator")
st.caption(
    "Upload multiple images. Smoothly traces strokes on a pure white canvas, "
    "blooms into vibrant ink, seamlessly merges into the original photograph, and holds it visibly before looping."
)

# ============================================================
# EXTRACTION & NOISE SUPPRESSION
# ============================================================

def clean_and_extract_elements(bgr_img):
    """
    Suppresses uneven lighting, shadows, and borders to isolate pure strokes.
    """
    h, w = bgr_img.shape[:2]
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

    # 1. Background illumination modeling
    kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    bg = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel_bg)
    diff = cv2.absdiff(bg, gray)
    norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # 2. Extract ink mask
    _, binary = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3. Clean up edge artifacts and border shadows
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_clean)

    border = max(10, int(min(h, w) * 0.02))
    binary[:border, :] = 0
    binary[-border:, :] = 0
    binary[:, :border] = 0
    binary[:, -border:] = 0

    # 4. Filter connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    min_area = max(50, int(h * w * 0.001))

    components = []
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        comp_mask = (labels == label).astype(np.uint8) * 255
        x = stats[label, cv2.CC_STAT_LEFT]
        y = stats[label, cv2.CC_STAT_TOP]
        cw = stats[label, cv2.CC_STAT_WIDTH]
        ch = stats[label, cv2.CC_STAT_HEIGHT]

        ink_pixels = bgr_img[comp_mask > 0]
        comp_bgr = np.median(ink_pixels, axis=0).astype(np.uint8)

        components.append({
            "id": len(components) + 1,
            "mask": comp_mask,
            "bbox": (x, y, x + cw, y + ch),
            "center": (centroids[label][0], centroids[label][1]),
            "area": area,
            "color": comp_bgr
        })

    # Sort left-to-right to mimic standard natural writing order
    components.sort(key=lambda c: c["bbox"][0])
    return components, binary

def compute_geodesic_progress_map(comp_mask, direction="Top -> Bottom"):
    """
    Generates a continuous progress gradient (0.0 to 1.0) along the stroke
    using geodesic distance propagation (BFS on pixel grid).
    """
    ys, xs = np.where(comp_mask > 0)
    if len(xs) == 0:
        return np.zeros_like(comp_mask, dtype=np.float32), []

    points = np.column_stack((ys, xs))
    if direction == "Top -> Bottom":
        start_idx = np.argmin(points[:, 0])
    elif direction == "Bottom -> Top":
        start_idx = np.argmax(points[:, 0])
    elif direction == "Left -> Right":
        start_idx = np.argmin(points[:, 1])
    else:  # Right -> Left
        start_idx = np.argmax(points[:, 1])

    start_pt = tuple(points[start_idx])

    h, w = comp_mask.shape
    dist_map = np.full((h, w), np.inf, dtype=np.float32)
    dist_map[start_pt] = 0.0

    queue = deque([start_pt])
    max_d = 0.0

    neighbors = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                 (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]

    while queue:
        cy, cx = queue.popleft()
        cd = dist_map[cy, cx]

        for dy, dx, weight in neighbors:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w and comp_mask[ny, nx] > 0:
                if cd + weight < dist_map[ny, nx]:
                    dist_map[ny, nx] = cd + weight
                    max_d = max(max_d, cd + weight)
                    queue.append((ny, nx))

    progress_map = np.zeros((h, w), dtype=np.float32)
    valid = comp_mask > 0
    if max_d > 0:
        progress_map[valid] = dist_map[valid] / max_d

    # Sample trajectory points for pen tip tracking
    sampled_path = []
    num_steps = 100
    for s in range(num_steps):
        target = s / float(num_steps - 1)
        sub_pts = points[np.abs(progress_map[points[:, 0], points[:, 1]] - target) < 0.05]
        if len(sub_pts) > 0:
            sampled_path.append((int(np.mean(sub_pts[:, 0])), int(np.mean(sub_pts[:, 1]))))
        elif sampled_path:
            sampled_path.append(sampled_path[-1])

    return progress_map, sampled_path

def enhance_vibrancy(bgr_img):
    """
    Enhances hue saturation and lighting contrast of the artwork.
    """
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.45, 0, 255)  # Saturation
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.15, 0, 255)  # Luminance
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

# ============================================================
# RENDERING PIPELINE
# ============================================================

def render_frame(
    original_bgr, enhanced_bgr, components, frame_idx, total_frames,
    write_ratio=0.60, glow_intensity=0.3, show_pen=True,
    enable_original_merge=True, merge_percent=20
):
    h, w = original_bgr.shape[:2]
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)

    t_global = frame_idx / float(max(1, total_frames - 1))
    write_phase_end = write_ratio

    merge_phase_start = 1.0 - (merge_percent / 100.0) if enable_original_merge else 1.0
    active_pen_point = None

    # --- Phase 1: Progressive Calligraphy Drawing ---
    t_write = min(1.0, t_global / write_phase_end)
    num_comps = len(components)

    for i, comp in enumerate(components):
        comp_start = i / float(num_comps)
        comp_end = (i + 1) / float(num_comps)

        if t_write < comp_start:
            continue

        comp_t = (t_write - comp_start) / (comp_end - comp_start)
        comp_t = max(0.0, min(1.0, comp_t))

        smooth_t = comp_t * comp_t * (3.0 - 2.0 * comp_t)

        pmap = comp["progress_map"]
        mask = comp["mask"]
        visible = mask & (pmap <= smooth_t + 0.03)

        edge_soft = cv2.GaussianBlur(visible.astype(np.uint8) * 255, (5, 5), 0) / 255.0
        for c in range(3):
            canvas[:, :, c] = np.clip(
                canvas[:, :, c] * (1.0 - edge_soft) + enhanced_bgr[:, :, c] * edge_soft,
                0, 255
            ).astype(np.uint8)

        if show_pen and 0.0 < comp_t < 1.0 and comp["path"]:
            p_idx = min(int(smooth_t * (len(comp["path"]) - 1)), len(comp["path"]) - 1)
            active_pen_point = comp["path"][p_idx]

    # --- Phase 2: Color Glow & Lightening ---
    if t_global > write_phase_end:
        glow_end = merge_phase_start if enable_original_merge else 1.0
        t_glow = min(1.0, (t_global - write_phase_end) / max(0.001, (glow_end - write_phase_end)))
        t_glow = t_glow * t_glow * (3.0 - 2.0 * t_glow)

        combined_mask = np.zeros((h, w), dtype=np.uint8)
        for comp in components:
            combined_mask = cv2.bitwise_or(combined_mask, comp["mask"])

        blur_k = max(15, (min(h, w) // 30) | 1)
        glow_bloom = cv2.GaussianBlur(enhanced_bgr, (blur_k, blur_k), 0).astype(np.float32)

        stroke_alpha = cv2.GaussianBlur(combined_mask, (7, 7), 0) / 255.0
        stroke_alpha = np.repeat(stroke_alpha[:, :, np.newaxis], 3, axis=2)

        bloomed = np.clip(
            enhanced_bgr.astype(np.float32) * (1.0 + glow_intensity * t_glow) +
            glow_bloom * (t_glow * 0.25),
            0, 255
        )

        canvas = np.clip(
            canvas.astype(np.float32) * (1.0 - stroke_alpha * t_glow) +
            bloomed * (stroke_alpha * t_glow),
            0, 255
        ).astype(np.uint8)

    # --- Pen Nib Indicator ---
    if show_pen and active_pen_point is not None and t_global <= write_phase_end:
        py, px = active_pen_point
        cv2.circle(canvas, (px, py), 4, (30, 30, 30), -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (px - 1, py - 1), 2, (230, 230, 230), -1, lineType=cv2.LINE_AA)

    # --- Phase 3: Smooth Merge into Original Image ---
    if enable_original_merge and t_global >= merge_phase_start:
        alpha = (t_global - merge_phase_start) / max(0.001, (1.0 - merge_phase_start))
        alpha = max(0.0, min(1.0, alpha))
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        canvas = cv2.addWeighted(canvas, 1.0 - alpha, original_bgr, alpha, 0)

    return canvas

def build_gif(frames, duration_ms=50):
    if not frames:
        return b""
    pil_frames = [f.convert("RGB").convert("P", palette=Image.ADAPTIVE, colors=256) for f in frames]
    buf = io.BytesIO()
    pil_frames[0].save(
        buf, format="GIF", save_all=True, append_images=pil_frames[1:],
        duration=duration_ms, loop=0, optimize=False
    )
    return buf.getvalue()

# ============================================================
# SIDEBAR CONTROLS
# ============================================================

st.sidebar.header("⚙️ Flow & Merge Settings")
write_speed = st.sidebar.slider("Writing Phase Share (%)", 30, 80, 55, help="Time spent drawing strokes before color bloom.")
glow_power = st.sidebar.slider("Color Lightening & Glow", 0.0, 0.6, 0.3, step=0.05)

enable_merge = st.sidebar.checkbox("Merge Back into Original Photo at End", value=True)
merge_duration = st.sidebar.slider("Transition Duration (%)", 5, 40, 20) if enable_merge else 0

st.sidebar.markdown("---")
st.sidebar.subheader("⏱️ Timing & Hold Settings")
total_animated_frames = st.sidebar.slider("Animation Transition Frames", 30, 120, 60, step=5)
frame_duration = st.sidebar.slider("Frame Delay (ms)", 20, 100, 45, step=5)
hold_seconds = st.sidebar.slider("Original Image Hold Duration (seconds)", 0.5, 4.0, 2.0, step=0.5,
                                help="How long the complete original image stays visible before restarting.")
show_pen = st.sidebar.checkbox("Show Brush Nib Tip", value=True)

# ============================================================
# MAIN APPLICATION
# ============================================================

uploaded_files = st.file_uploader(
    "Upload Calligraphy Images (Select multiple files at once)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    st.info(f"Loaded **{len(uploaded_files)}** image(s). Customize stroke directions or render animations below.")
    tabs = st.tabs([f"📄 {f.name}" for f in uploaded_files])

    cached_data = {}

    for tab, file in zip(tabs, uploaded_files):
        with tab:
            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            max_size = 900
            h, w = img.shape[:2]
            if max(h, w) > max_size:
                scale = max_size / float(max(h, w))
                img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

            components, clean_mask = clean_and_extract_elements(img)
            enhanced = enhance_vibrancy(img)

            c1, c2 = st.columns([1, 1])
            with c1:
                st.subheader("Cleaned Ink Isolation")
                st.image(cv2.cvtColor(clean_mask, cv2.COLOR_GRAY2RGB), caption="Extracted strokes without paper borders", use_container_width=True)

            with c2:
                st.subheader("Stroke Configurations")
                for i, comp in enumerate(components):
                    col_dir = st.selectbox(
                        f"Stroke {i+1} Direction",
                        ["Top -> Bottom", "Left -> Right", "Bottom -> Top", "Right -> Left"],
                        index=0,
                        key=f"dir_{file.name}_{i}"
                    )
                    pmap, path = compute_geodesic_progress_map(comp["mask"], col_dir)
                    comp["progress_map"] = pmap
                    comp["path"] = path

            cached_data[file.name] = {
                "original": img,
                "enhanced": enhanced,
                "components": components
            }

    st.markdown("---")
    if st.button("🚀 Render All Smooth Animations (.ZIP)", type="primary"):
        prog = st.progress(0)
        status = st.empty()
        zip_buf = io.BytesIO()

        # Calculate exact number of hold frames based on selected seconds and frame duration
        hold_frames_count = int((hold_seconds * 1000) / frame_duration)

        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for idx, file in enumerate(uploaded_files):
                status.write(f"Rendering ({idx + 1}/{len(uploaded_files)}): **{file.name}**...")
                item = cached_data[file.name]

                frames = []

                # Render dynamic drawing, blooming, and merge transition
                for f_idx in range(total_animated_frames):
                    rendered = render_frame(
                        item["original"],
                        item["enhanced"],
                        item["components"],
                        f_idx,
                        total_animated_frames,
                        write_ratio=write_speed / 100.0,
                        glow_intensity=glow_power,
                        show_pen=show_pen,
                        enable_original_merge=enable_merge,
                        merge_percent=merge_duration
                    )
                    frames.append(Image.fromarray(cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)))

                # Hold the final original image frame
                final_frame = frames[-1]
                for _ in range(hold_frames_count):
                    frames.append(final_frame)

                gif_bytes = build_gif(frames, duration_ms=frame_duration)
                zf.writestr(f"animated_{file.name.rsplit('.', 1)[0]}.gif", gif_bytes)
                prog.progress((idx + 1) / len(uploaded_files))

        status.success("🎉 All animations generated successfully with end pause!")
        prog.empty()

        zip_buf.seek(0)
        st.download_button(
            "📦 Download Smooth Animated GIFs (.ZIP)",
            data=zip_buf.getvalue(),
            file_name="calligraphy_smooth_animations.zip",
            mime="application/zip"
        )
