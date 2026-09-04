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
    page_title="Cinematic 3D Calligraphy Animator",
    page_icon="✒️",
    layout="wide",
)

st.title("✒️ Cinematic 3D Calligraphy Flow & Finish Animator")
st.caption(
    "Traces strokes naturally on white canvas, merges into the original photograph, "
    "and renders a high-end 3D material finish (Metallic Foil, Glossy Wet Resin, or Letterpress) with advanced studio lighting."
)

# ============================================================
# EXTRACTION & NOISE SUPPRESSION
# ============================================================

def clean_and_extract_elements(bgr_img):
    h, w = bgr_img.shape[:2]
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

    # Illumination background correction
    kernel_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    bg = cv2.morphologyEx(gray, cv2.MORPH_DILATE, kernel_bg)
    diff = cv2.absdiff(bg, gray)
    norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    _, binary = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_clean)

    # Border vignette clearing
    border = max(10, int(min(h, w) * 0.02))
    binary[:border, :] = 0
    binary[-border:, :] = 0
    binary[:, :border] = 0
    binary[:, -border:] = 0

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

    components.sort(key=lambda c: c["bbox"][0])
    return components, binary

def compute_geodesic_progress_map(comp_mask, direction="Top -> Bottom"):
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
    else:
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

def apply_cinematic_color_correction(bgr_img):
    """
    Subtle S-curve contrast balance, gentle saturation lift, and clean highlights.
    """
    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR).astype(np.float32)

    # Slight saturation boost without burning
    hsv = cv2.cvtColor(enhanced_bgr.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.25, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

# ============================================================
# MULTI-STYLE 3D ENGINE
# ============================================================

def render_3d_style(base_img, mask, style="Glossy Wet Resin", depth=1.0, light_angle=45):
    h, w = mask.shape
    dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    max_d = np.max(dist)
    if max_d <= 0:
        return base_img

    norm_dist = np.clip(dist / min(22.0, max_d), 0.0, 1.0)
    smooth_mask = cv2.GaussianBlur((mask > 0).astype(np.float32), (5, 5), 0)

    # Light vectors
    rad = math.radians(light_angle)
    lx, ly = math.cos(rad), -math.sin(rad)
    lz = 0.85
    l_len = math.sqrt(lx * lx + ly * ly + lz * lz)
    lx, ly, lz = lx / l_len, ly / l_len, lz / l_len

    out = base_img.astype(np.float32).copy()

    if style == "Glossy Wet Resin":
        # Domed dome profile with glossy specular sheen
        height = np.sqrt(np.clip(norm_dist, 0.0, 1.0)) * depth
        height = cv2.GaussianBlur(height, (5, 5), 0)

        gx = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3)
        nz = np.ones_like(gx)
        n_len = np.sqrt(gx**2 + gy**2 + nz**2)
        nx, ny, nz = -gx / n_len, -gy / n_len, nz / n_len

        # Diffuse & sharp liquid specular highlight
        diff = np.clip(nx * lx + ny * ly + nz * lz, 0.0, 1.0)
        view_z = 1.0
        hx, hy, hz = lx, ly, lz + view_z
        hlen = math.sqrt(hx*hx + hy*hy + hz*hz)
        hx, hy, hz = hx / hlen, hy / hlen, hz / hlen
        spec = np.clip(nx * hx + ny * hy + nz * hz, 0.0, 1.0) ** 28 * (height > 0.08)

        # Ambient drop shadow
        m_sh = np.float32([[1, 0, int(lx * 4 * depth)], [0, 1, int(-ly * 4 * depth)]])
        sh_mask = cv2.warpAffine((mask > 0).astype(np.float32), m_sh, (w, h))
        sh_soft = cv2.GaussianBlur(np.clip(sh_mask - (mask > 0), 0.0, 1.0), (11, 11), 0)

        for c in range(3):
            out[:, :, c] *= (1.0 - sh_soft * 0.32)
            c_val = base_img[:, :, c].astype(np.float32)
            shaded = c_val * (0.80 + 0.35 * diff) + (spec * 130.0)
            out[:, :, c] = out[:, :, c] * (1.0 - smooth_mask) + shaded * smooth_mask

    elif style == "Sleek Metallic Foil":
        # Chiseled bevel profile with directional metal luster
        height = cv2.GaussianBlur(norm_dist, (3, 3), 0) * depth
        gx = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3)
        nz = np.ones_like(gx)
        n_len = np.sqrt(gx**2 + gy**2 + nz**2)
        nx, ny, nz = -gx / n_len, -gy / n_len, nz / n_len

        diff = np.clip(nx * lx + ny * ly + nz * lz, 0.0, 1.0)
        metal_luster = np.sin(diff * math.pi) ** 2

        # Secondary rim light
        rim = np.clip(1.0 - nz, 0.0, 1.0) ** 2 * (height > 0.05)

        for c in range(3):
            c_val = base_img[:, :, c].astype(np.float32)
            foil = c_val * (0.65 + 0.65 * metal_luster) + (rim * 80.0)
            out[:, :, c] = out[:, :, c] * (1.0 - smooth_mask) + foil * smooth_mask

    else:  # Deep Letterpress (Debossed Intaglio)
        # Indented inverse bevel
        height = cv2.GaussianBlur(norm_dist, (7, 7), 0) * depth
        gx = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3)
        nz = np.ones_like(gx)
        n_len = np.sqrt(gx**2 + gy**2 + nz**2)
        nx, ny, nz = -gx / n_len, -gy / n_len, nz / n_len

        # Inward crevice shadow & inner rim light
        inner_shadow = np.clip(-(nx * lx + ny * ly), 0.0, 1.0) * (height > 0.05)
        inner_rim = np.clip(nx * lx + ny * ly, 0.0, 1.0) * (height > 0.05)

        for c in range(3):
            c_val = base_img[:, :, c].astype(np.float32)
            pressed = c_val * (1.0 - inner_shadow * 0.45) + (inner_rim * 40.0)
            out[:, :, c] = out[:, :, c] * (1.0 - smooth_mask) + pressed * smooth_mask

    return np.clip(out, 0, 255).astype(np.uint8)

# ============================================================
# RENDERING PIPELINE
# ============================================================

def render_frame(
    original_bgr, enhanced_bgr, components, clean_mask, frame_idx, total_frames,
    write_ratio=0.55, glow_ratio=0.15, merge_ratio=0.15,
    style_3d="Glossy Wet Resin", show_pen=True
):
    h, w = original_bgr.shape[:2]
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)

    t_global = frame_idx / float(max(1, total_frames - 1))
    t_end_write = write_ratio
    t_end_glow = t_end_write + glow_ratio
    t_end_merge = t_end_glow + merge_ratio

    active_pen_point = None

    # Phase 1: Progressive Calligraphy Tracing
    t_write = min(1.0, t_global / max(0.001, t_end_write))
    num_comps = len(components)

    for i, comp in enumerate(components):
        comp_start = i / float(num_comps)
        comp_end = (i + 1) / float(num_comps)

        if t_write < comp_start:
            continue

        comp_t = (t_write - comp_start) / max(0.001, (comp_end - comp_start))
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

    # Phase 2: Vibrant Glow
    if t_global > t_end_write:
        t_glow = min(1.0, (t_global - t_end_write) / max(0.001, (t_end_glow - t_end_write)))
        t_glow = t_glow * t_glow * (3.0 - 2.0 * t_glow)

        blur_k = max(15, (min(h, w) // 30) | 1)
        glow_bloom = cv2.GaussianBlur(enhanced_bgr, (blur_k, blur_k), 0).astype(np.float32)

        stroke_alpha = cv2.GaussianBlur(clean_mask, (7, 7), 0) / 255.0
        stroke_alpha = np.repeat(stroke_alpha[:, :, np.newaxis], 3, axis=2)

        bloomed = np.clip(
            enhanced_bgr.astype(np.float32) * (1.0 + 0.25 * t_glow) +
            glow_bloom * (t_glow * 0.20),
            0, 255
        )
        canvas = np.clip(
            canvas.astype(np.float32) * (1.0 - stroke_alpha * t_glow) +
            bloomed * (stroke_alpha * t_glow),
            0, 255
        ).astype(np.uint8)

    # Pen Tip Indicator
    if show_pen and active_pen_point is not None and t_global <= t_end_write:
        py, px = active_pen_point
        cv2.circle(canvas, (px, py), 4, (30, 30, 30), -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (px - 1, py - 1), 2, (230, 230, 230), -1, lineType=cv2.LINE_AA)

    # Phase 3: Transition to Original Image
    if t_global >= t_end_glow:
        t_merge = min(1.0, (t_global - t_end_glow) / max(0.001, (t_end_merge - t_end_glow)))
        t_merge = t_merge * t_merge * (3.0 - 2.0 * t_merge)
        canvas = cv2.addWeighted(canvas, 1.0 - t_merge, original_bgr, t_merge, 0)

    # Phase 4: Clean 3D Material Transformation
    if t_global >= t_end_merge:
        t_3d = min(1.0, (t_global - t_end_merge) / max(0.001, (1.0 - t_end_merge)))
        t_3d = t_3d * t_3d * (3.0 - 2.0 * t_3d)

        frame_3d = render_3d_style(original_bgr, clean_mask, style=style_3d, depth=t_3d)
        canvas = cv2.addWeighted(canvas, 1.0 - t_3d, frame_3d, t_3d, 0)

    return canvas

def build_gif(frames, duration_ms=45):
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
# INTERFACE CONTROLS
# ============================================================

st.sidebar.header("🎨 Visual 3D Styling")
selected_3d_style = st.sidebar.selectbox(
    "Select 3D Material Finish",
    ["Glossy Wet Resin", "Sleek Metallic Foil", "Deep Letterpress (Deboss)"],
    index=0,
    help="Realistic finishes using clean physical lighting rather than noisy colors."
)

st.sidebar.markdown("---")
st.sidebar.subheader("⏱️ Sequence & Timings")
write_pct = st.sidebar.slider("Writing Time (%)", 35, 65, 45)
total_animated_frames = st.sidebar.slider("Animation Transition Frames", 40, 120, 65, step=5)
frame_duration = st.sidebar.slider("Frame Delay (ms)", 25, 80, 45, step=5)
hold_seconds = st.sidebar.slider("Final 3D Hold Duration (seconds)", 0.5, 4.0, 2.0, step=0.5)
show_pen = st.sidebar.checkbox("Show Brush Nib Tip", value=True)

uploaded_files = st.file_uploader(
    "Upload Calligraphy Images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    st.info(f"Loaded **{len(uploaded_files)}** file(s). Set stroke orientations below or render animations.")
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
            enhanced = apply_cinematic_color_correction(img)

            c1, c2 = st.columns([1, 1])
            with c1:
                st.subheader("Extracted Calligraphy Ink")
                st.image(cv2.cvtColor(clean_mask, cv2.COLOR_GRAY2RGB), caption="Clean stroke mask", use_container_width=True)

            with c2:
                st.subheader("Stroke Configurations")
                for i, comp in enumerate(components):
                    col_dir = st.selectbox(
                        f"Stroke {i+1} Writing Direction",
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
                "components": components,
                "clean_mask": clean_mask
            }

    st.markdown("---")
    if st.button("🚀 Render All 3D Finished Animations (.ZIP)", type="primary"):
        prog = st.progress(0)
        status = st.empty()
        zip_buf = io.BytesIO()

        hold_frames_count = int((hold_seconds * 1000) / frame_duration)

        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for idx, file in enumerate(uploaded_files):
                status.write(f"Processing ({idx + 1}/{len(uploaded_files)}): **{file.name}**...")
                item = cached_data[file.name]

                frames = []
                w_ratio = write_pct / 100.0

                for f_idx in range(total_animated_frames):
                    rendered = render_frame(
                        item["original"],
                        item["enhanced"],
                        item["components"],
                        item["clean_mask"],
                        f_idx,
                        total_animated_frames,
                        write_ratio=w_ratio,
                        glow_ratio=0.15,
                        merge_ratio=0.15,
                        style_3d=selected_3d_style,
                        show_pen=show_pen
                    )
                    frames.append(Image.fromarray(cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)))

                # Hold the final finished 3D frame
                final_frame = frames[-1]
                for _ in range(hold_frames_count):
                    frames.append(final_frame)

                gif_bytes = build_gif(frames, duration_ms=frame_duration)
                zf.writestr(f"animated_{file.name.rsplit('.', 1)[0]}.gif", gif_bytes)
                prog.progress((idx + 1) / len(uploaded_files))

        status.success("🎉 Complete! Calligraphy rendered with selected 3D finish.")
        prog.empty()

        zip_buf.seek(0)
        st.download_button(
            "📦 Download High-End 3D Animated GIFs (.ZIP)",
            data=zip_buf.getvalue(),
            file_name="calligraphy_high_end_3d.zip",
            mime="application/zip"
        )
