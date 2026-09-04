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
    page_title="Smooth Calligraphy Flow & 3D Emboss Animator",
    page_icon="✒️",
    layout="wide",
)

st.title("✒️ Calligraphy Flow, Merge & 3D Emboss Animator")
st.caption(
    "Traces strokes smoothly on white, blooms colors, merges into the original photograph, "
    "and finally extrudes the calligraphy into an elegant, clean 3D beveled relief object."
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

    # Binarization
    _, binary = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Edge and border suppression
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_clean)

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

def enhance_vibrancy(bgr_img):
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.45, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.15, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

# ============================================================
# CLEAN 3D RELIEF PIPELINE (NO CRAZY COLORS / MOTION)
# ============================================================

def generate_3d_relief(base_img, combined_mask, depth_factor=1.0, light_angle=45):
    """
    Constructs a clean, non-distorting 3D beveled relief with ambient occlusion,
    diffuse lighting, and subtle specular highlight using the real ink colors.
    """
    h, w = combined_mask.shape

    # Inner elevation map using distance transform
    dist = cv2.distanceTransform((combined_mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    max_d = np.max(dist)
    if max_d > 0:
        height_map = np.clip(dist / min(18.0, max_d), 0.0, 1.0)
    else:
        height_map = np.zeros((h, w), dtype=np.float32)

    # Smooth the bevel gradient
    height_map = cv2.GaussianBlur(height_map, (7, 7), 0) * depth_factor

    # Compute surface normal gradients
    grad_x = cv2.Sobel(height_map, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(height_map, cv2.CV_32F, 0, 1, ksize=3)

    # Studio light vector (fixed direction from top-left)
    rad = math.radians(light_angle)
    lx, ly, lz = math.cos(rad), -math.sin(rad), 0.75
    l_len = math.sqrt(lx * lx + ly * ly + lz * lz)
    lx, ly, lz = lx / l_len, ly / l_len, lz / l_len

    # Surface normals
    nz = np.ones_like(grad_x)
    n_len = np.sqrt(grad_x**2 + grad_y**2 + nz**2)
    nx = -grad_x / n_len
    ny = -grad_y / n_len
    nz = nz / n_len

    # Diffuse shading (Lambertian)
    diffuse = np.clip(nx * lx + ny * ly + nz * lz, 0.0, 1.0)

    # Specular bevel highlight (clean white sheen on the rim)
    view_z = 1.0
    half_x, half_y, half_z = lx, ly, lz + view_z
    h_len = np.sqrt(half_x**2 + half_y**2 + half_z**2)
    half_x, half_y, half_z = half_x / h_len, half_y / h_len, half_z / h_len

    specular = np.clip(nx * half_x + ny * half_y + nz * half_z, 0.0, 1.0) ** 16
    specular *= (height_map > 0.05).astype(np.float32)

    # Ambient drop shadow under the 3D stroke
    shadow_shift_x = int(round(lx * 4 * depth_factor))
    shadow_shift_y = int(round(-ly * 4 * depth_factor))
    m = np.float32([[1, 0, shadow_shift_x], [0, 1, shadow_shift_y]])
    shadow_mask = cv2.warpAffine((combined_mask > 0).astype(np.float32), m, (w, h))
    shadow_soft = cv2.GaussianBlur(shadow_mask, (15, 15), 0)
    shadow_soft = np.clip(shadow_soft - (combined_mask > 0).astype(np.float32), 0.0, 1.0)

    # Compose output on the real image
    rendered = base_img.astype(np.float32)

    # Darken paper behind drop shadow
    for c in range(3):
        rendered[:, :, c] *= (1.0 - shadow_soft * 0.35)

    # Apply 3D bevel lighting to strokes
    mask_soft = cv2.GaussianBlur((combined_mask > 0).astype(np.float32), (5, 5), 0)
    for c in range(3):
        stroke_color = base_img[:, :, c].astype(np.float32)
        shaded_stroke = stroke_color * (0.65 + 0.5 * diffuse) + (specular * 80.0)
        rendered[:, :, c] = rendered[:, :, c] * (1.0 - mask_soft) + shaded_stroke * mask_soft

    return np.clip(rendered, 0, 255).astype(np.uint8)

# ============================================================
# RENDERING PIPELINE
# ============================================================

def render_frame(
    original_bgr, enhanced_bgr, components, combined_mask, frame_idx, total_frames,
    write_ratio=0.50, glow_ratio=0.15, merge_ratio=0.15, show_pen=True
):
    """
    Phases:
    1. Progressive Drawing on White Canvas
    2. Color Glow & Bloom
    3. Smooth Merge into Original Image
    4. Transformation into 3D Relief Object
    """
    h, w = original_bgr.shape[:2]
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)

    t_global = frame_idx / float(max(1, total_frames - 1))

    # Phase breakpoints
    t_end_write = write_ratio
    t_end_glow = t_end_write + glow_ratio
    t_end_merge = t_end_glow + merge_ratio

    active_pen_point = None

    # --- Phase 1: Progressive Calligraphy Drawing ---
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

    # --- Phase 2: Color Glow & Lightening ---
    if t_global > t_end_write:
        t_glow = min(1.0, (t_global - t_end_write) / max(0.001, (t_end_glow - t_end_write)))
        t_glow = t_glow * t_glow * (3.0 - 2.0 * t_glow)

        blur_k = max(15, (min(h, w) // 30) | 1)
        glow_bloom = cv2.GaussianBlur(enhanced_bgr, (blur_k, blur_k), 0).astype(np.float32)

        stroke_alpha = cv2.GaussianBlur(combined_mask, (7, 7), 0) / 255.0
        stroke_alpha = np.repeat(stroke_alpha[:, :, np.newaxis], 3, axis=2)

        bloomed = np.clip(
            enhanced_bgr.astype(np.float32) * (1.0 + 0.3 * t_glow) +
            glow_bloom * (t_glow * 0.25),
            0, 255
        )

        canvas = np.clip(
            canvas.astype(np.float32) * (1.0 - stroke_alpha * t_glow) +
            bloomed * (stroke_alpha * t_glow),
            0, 255
        ).astype(np.uint8)

    # Brush nib tip
    if show_pen and active_pen_point is not None and t_global <= t_end_write:
        py, px = active_pen_point
        cv2.circle(canvas, (px, py), 4, (30, 30, 30), -1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, (px - 1, py - 1), 2, (230, 230, 230), -1, lineType=cv2.LINE_AA)

    # --- Phase 3: Smooth Merge into Original Image ---
    if t_global >= t_end_glow:
        t_merge = min(1.0, (t_global - t_end_glow) / max(0.001, (t_end_merge - t_end_glow)))
        t_merge = t_merge * t_merge * (3.0 - 2.0 * t_merge)
        canvas = cv2.addWeighted(canvas, 1.0 - t_merge, original_bgr, t_merge, 0)

    # --- Phase 4: Clean 3D Emboss Relief ---
    if t_global >= t_end_merge:
        t_3d = min(1.0, (t_global - t_end_merge) / max(0.001, (1.0 - t_end_merge)))
        t_3d = t_3d * t_3d * (3.0 - 2.0 * t_3d)

        # Extrude depth gently (without extreme rotation or color noise)
        relief_frame = generate_3d_relief(original_bgr, combined_mask, depth_factor=t_3d)

        # Subtle slight camera tilt (micro perspective shift of 1-2 pixels)
        tilt_dx = int(round(1.5 * t_3d))
        tilt_dy = int(round(-1.0 * t_3d))
        m_tilt = np.float32([[1, 0, tilt_dx], [0, 1, tilt_dy]])
        relief_frame = cv2.warpAffine(relief_frame, m_tilt, (w, h), borderMode=cv2.BORDER_REPLICATE)

        canvas = cv2.addWeighted(canvas, 1.0 - t_3d, relief_frame, t_3d, 0)

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

st.sidebar.header("⚙️ Sequence & 3D Controls")
write_pct = st.sidebar.slider("Writing Time (%)", 30, 65, 45)
total_animated_frames = st.sidebar.slider("Animation Transition Frames", 40, 120, 65, step=5)
frame_duration = st.sidebar.slider("Frame Delay (ms)", 25, 90, 45, step=5)
hold_seconds = st.sidebar.slider("Final 3D Hold Duration (seconds)", 0.5, 4.0, 2.0, step=0.5)
show_pen = st.sidebar.checkbox("Show Brush Nib Tip", value=True)

# ============================================================
# MAIN APPLICATION
# ============================================================

uploaded_files = st.file_uploader(
    "Upload Calligraphy Images (Multiple Files Supported)",
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
    if st.button("🚀 Render All 3D Relief Animations (.ZIP)", type="primary"):
        prog = st.progress(0)
        status = st.empty()
        zip_buf = io.BytesIO()

        hold_frames_count = int((hold_seconds * 1000) / frame_duration)

        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for idx, file in enumerate(uploaded_files):
                status.write(f"Animating ({idx + 1}/{len(uploaded_files)}): **{file.name}**...")
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
                        show_pen=show_pen
                    )
                    frames.append(Image.fromarray(cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)))

                # Hold the final clean 3D relief
                final_frame = frames[-1]
                for _ in range(hold_frames_count):
                    frames.append(final_frame)

                gif_bytes = build_gif(frames, duration_ms=frame_duration)
                zf.writestr(f"animated_{file.name.rsplit('.', 1)[0]}.gif", gif_bytes)
                prog.progress((idx + 1) / len(uploaded_files))

        status.success("🎉 Complete! Calligraphy written, merged, and rendered into 3D relief.")
        prog.empty()

        zip_buf.seek(0)
        st.download_button(
            "📦 Download Animated 3D Relief GIFs (.ZIP)",
            data=zip_buf.getvalue(),
            file_name="calligraphy_3d_animations.zip",
            mime="application/zip"
        )
