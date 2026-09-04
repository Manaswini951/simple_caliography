import gc
import io
import math
import zipfile

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Artistic Calligraphy Animator",
    page_icon="✒️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("✒️ Artistic Calligraphy Writing Animator")

st.write(
    "Upload calligraphy artwork to automatically clean the background, "
    "extract ink strokes, and create smooth fluid-writing animations."
)

# ============================================================
# CONSTANTS & CONFIG
# ============================================================

MAX_IMAGE_DIM = 500
MIN_COMPONENT_AREA = 25
MAX_SKELETON_POINTS = 8000

# ============================================================
# IMAGE & PREPROCESSING UTILITIES
# ============================================================

def resize_if_large(img, max_dim=MAX_IMAGE_DIM):
    if img is None:
        return None
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img
    scale = max_dim / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def decode_image(file_bytes):
    try:
        array = np.frombuffer(file_bytes, dtype=np.uint8)
        return cv2.imdecode(array, cv2.IMREAD_COLOR)
    except Exception:
        return None


def preprocess_and_clean_background(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bg_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    bg = cv2.dilate(gray, bg_kernel)
    bg = cv2.medianBlur(bg, 15)

    diff = cv2.absdiff(gray, bg)
    norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    _, binary = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned_binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned_binary = cv2.morphologyEx(cleaned_binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    return norm, cleaned_binary


def extract_stroke_components(binary_mask, min_area=MIN_COMPONENT_AREA):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    components = []

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])

        component_mask = labels == label
        components.append({
            "id": len(components) + 1,
            "mask": component_mask,
            "bbox": (x, y, x + width - 1, y + height - 1),
            "center": (float(centroids[label][0]), float(centroids[label][1])),
            "area": area,
        })

    components.sort(key=lambda c: (c["bbox"][1] // 20, c["bbox"][0]))

    for i, component in enumerate(components, start=1):
        component["id"] = i

    return components


# ============================================================
# SKELETON & PATH PROCESSING
# ============================================================

def morphological_skeleton(binary):
    binary_bytes = (binary.astype(np.uint8) * 255)
    skeleton = np.zeros_like(binary_bytes)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    current = binary_bytes.copy()
    iterations = 0

    while True:
        eroded = cv2.erode(current, element)
        opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(eroded, opened)
        skeleton = cv2.bitwise_or(skeleton, temp)
        current = eroded
        iterations += 1

        if cv2.countNonZero(current) == 0 or iterations > 500:
            break

    return skeleton > 0


def generate_ordered_path(skeleton, direction="Left -> Right"):
    ys, xs = np.where(skeleton)
    if len(xs) == 0:
        return []

    if len(xs) > MAX_SKELETON_POINTS:
        step = int(math.ceil(len(xs) / float(MAX_SKELETON_POINTS)))
        ys = ys[::step]
        xs = xs[::step]

    points = list(zip(ys.astype(int), xs.astype(int)))
    if not points:
        return []

    if direction == "Top -> Bottom":
        start = min(points, key=lambda p: (p[0], p[1]))
    elif direction == "Right -> Left":
        start = max(points, key=lambda p: (p[1], -p[0]))
    else:
        start = min(points, key=lambda p: (p[1], p[0]))

    remaining = set(points)
    if start in remaining:
        remaining.remove(start)

    path = [start]
    current = start
    max_len = min(len(points), MAX_SKELETON_POINTS)

    while remaining and len(path) < max_len:
        cy, cx = current
        nearest = min(remaining, key=lambda p: ((p[0] - cy) ** 2 + (p[1] - cx) ** 2))
        dist_sq = (nearest[0] - cy) ** 2 + (nearest[1] - cx) ** 2

        if dist_sq > 1600:
            break

        path.append(nearest)
        remaining.remove(nearest)
        current = nearest

    return path


def smooth_stroke_path(path, window_size=5):
    if len(path) < 3:
        return path

    pts = np.array([[p[1], p[0]] for p in path], dtype=np.float32)
    kernel = np.ones(window_size, dtype=np.float32) / float(window_size)

    smooth_x = np.convolve(pts[:, 0], kernel, mode="same")
    smooth_y = np.convolve(pts[:, 1], kernel, mode="same")

    half = window_size // 2
    smooth_x[:half], smooth_x[-half:] = pts[:half, 0], pts[-half:, 0]
    smooth_y[:half], smooth_y[-half:] = pts[:half, 1], pts[-half:, 1]

    return [(int(round(y)), int(round(x))) for x, y in zip(smooth_x, smooth_y)]


def prepare_stroke_progress_map(component, direction):
    x1, y1, x2, y2 = component["bbox"]
    local_mask = component["mask"][y1:y2 + 1, x1:x2 + 1]

    skeleton = morphological_skeleton(local_mask)
    local_path = generate_ordered_path(skeleton, direction)
    local_path = smooth_stroke_path(local_path)

    global_path = [(y + y1, x + x1) for y, x in local_path]
    full_h, full_w = component["mask"].shape
    progress_map = np.zeros((full_h, full_w), dtype=np.float32)

    if not global_path or len(global_path) <= 1:
        progress_map[component["mask"]] = 1.0
        return global_path, progress_map

    local_h, local_w = local_mask.shape
    path_mask = np.ones((local_h, local_w), dtype=np.uint8)

    for py, px in local_path:
        ly, lx = py - y1, px - x1
        if 0 <= ly < local_h and 0 <= lx < local_w:
            path_mask[ly, lx] = 0

    _, labels = cv2.distanceTransformWithLabels(
        path_mask, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
    )

    label_to_progress = {}
    for idx, (py, px) in enumerate(local_path):
        ly, lx = py - y1, px - x1
        if 0 <= ly < local_h and 0 <= lx < local_w:
            lbl = int(labels[ly, lx])
            if lbl > 0:
                label_to_progress[lbl] = idx

    component_labels = labels[local_mask]
    if component_labels.size:
        values = np.zeros(component_labels.shape, dtype=np.float32)
        path_len = float(max(1, len(local_path) - 1))
        unique_labels = np.unique(component_labels)

        for lbl in unique_labels:
            lbl_int = int(lbl)
            prog_idx = label_to_progress.get(lbl_int, len(local_path) - 1)
            values[component_labels == lbl_int] = prog_idx / path_len

        progress_map[y1:y2 + 1, x1:x2 + 1][local_mask] = values

    return global_path, progress_map


# ============================================================
# RENDERING & EFX
# ============================================================

def ease_in_out(t):
    t = max(0.0, min(1.0, float(t)))
    return t * t * (3.0 - 2.0 * t)


def render_artistic_nib(canvas, point, angle=45, nib_size=6, color=(20, 20, 20)):
    if point is None:
        return
    y, x = point
    rad = math.radians(angle)
    dx = int(round((nib_size / 2.0) * math.cos(rad)))
    dy = int(round((nib_size / 2.0) * math.sin(rad)))
    cv2.line(canvas, (x - dx, y - dy), (x + dx, y + dy), color, 2, lineType=cv2.LINE_AA)


def render_frame(original_img, clean_bg, components, global_progress, show_nib=True):
    canvas = clean_bg.copy()
    num_components = len(components)
    if num_components == 0:
        return canvas

    active_nib_point = None

    for idx, component in enumerate(components):
        start_t = idx / float(num_components)
        end_t = (idx + 1) / float(num_components)

        if global_progress < start_t:
            continue

        local_progress = (global_progress - start_t) / (end_t - start_t) if end_t > start_t else 1.0
        local_progress = ease_in_out(local_progress)

        mask = component["mask"]
        progress_map = component["progress_map"]
        reveal_threshold = local_progress + 0.025

        active_pixels = mask & (progress_map <= reveal_threshold)
        canvas[active_pixels] = original_img[active_pixels]

        if show_nib and 0.0 < local_progress < 1.0:
            path = component.get("path", [])
            if path:
                p_idx = max(0, min(int(round(local_progress * (len(path) - 1))), len(path) - 1))
                active_nib_point = path[p_idx]

    if show_nib and active_nib_point is not None:
        render_artistic_nib(canvas, active_nib_point)

    return canvas


def build_gif(frames, duration=50):
    if not frames:
        return b""
    prepared = []
    try:
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(rgb).convert("P", palette=Image.ADAPTIVE, colors=128)
            prepared.append(pil_frame)

        buffer = io.BytesIO()
        prepared[0].save(
            buffer,
            format="GIF",
            save_all=True,
            append_images=prepared[1:],
            duration=int(duration),
            loop=0,
            optimize=True,
        )
        return buffer.getvalue()
    except Exception:
        return b""
    finally:
        prepared.clear()
        gc.collect()


# ============================================================
# MAIN APPLICATION INTERFACE
# ============================================================

st.sidebar.header("🎨 Global Writing Settings")
canvas_style = st.sidebar.selectbox("Background Style", ["Pure White Canvas", "Cleaned Paper Texture", "Original Image"])
writing_direction = st.sidebar.selectbox("Default Writing Direction", ["Left -> Right", "Top -> Bottom", "Right -> Left"])
show_nib = st.sidebar.checkbox("Show Calligraphy Pen Tip", value=True)
total_frames = st.sidebar.slider("Total Frames per GIF", min_value=30, max_value=90, value=45, step=5)
gif_speed = st.sidebar.slider("Frame Delay (ms)", min_value=20, max_value=100, value=40, step=5)

uploaded_files = st.file_uploader(
    "Upload Calligraphy Images (Multiple Selection Supported)",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.info(f"📁 {len(uploaded_files)} file(s) uploaded.")
    tabs = st.tabs([f"🖼️ {file.name}" for file in uploaded_files])
    processed_data = {}

    for tab, file in zip(tabs, uploaded_files):
        with tab:
            file.seek(0)
            file_bytes = file.read()
            if not file_bytes:
                st.error(f"❌ `{file.name}` is empty.")
                continue

            image = decode_image(file_bytes)
            image = resize_if_large(image, MAX_IMAGE_DIM)

            if image is None:
                st.error(f"❌ Could not decode `{file.name}`.")
                continue

            norm_img, binary_mask = preprocess_and_clean_background(image)
            components = extract_stroke_components(binary_mask)

            if not components:
                st.warning(f"⚠️ No ink detected in `{file.name}`.")
                continue

            for comp in components:
                path, progress_map = prepare_stroke_progress_map(comp, writing_direction)
                comp["path"] = path
                comp["progress_map"] = progress_map

            if canvas_style == "Pure White Canvas":
                base_bg = np.full_like(image, 255, dtype=np.uint8)
            elif canvas_style == "Cleaned Paper Texture":
                base_bg = cv2.cvtColor(norm_img, cv2.COLOR_GRAY2BGR)
                base_bg = cv2.normalize(base_bg, None, 215, 255, cv2.NORM_MINMAX)
            else:
                base_bg = image.copy()

            col1, col2 = st.columns(2)
            with col1:
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_container_width=True)
            with col2:
                st.image(binary_mask, caption=f"Extracted Clean Strokes ({len(components)} strokes)", use_container_width=True)

            processed_data[file.name] = {
                "image": image,
                "base_bg": base_bg,
                "components": components,
            }

    st.markdown("---")

    if st.button("🚀 Render All Animations & Download ZIP", type="primary", use_container_width=True):
        if not processed_data:
            st.error("❌ No valid images to render.")
        else:
            batch_progress = st.progress(0.0)
            status_text = st.empty()
            zip_buffer = io.BytesIO()

            valid_files = [f for f in uploaded_files if f.name in processed_data]
            total_valid = len(valid_files)

            try:
                with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
                    for idx, file in enumerate(valid_files):
                        status_text.write(f"🎬 Rendering **{file.name}** ({idx + 1}/{total_valid})...")
                        item = processed_data[file.name]
                        frames = []

                        for frame_index in range(total_frames):
                            progress = frame_index / float(max(1, total_frames - 1))
                            frame = render_frame(item["image"], item["base_bg"], item["components"], progress, show_nib=show_nib)
                            frames.append(frame)

                        gif_bytes = build_gif(frames, duration=gif_speed)
                        frames.clear()
                        gc.collect()

                        if gif_bytes:
                            clean_name = file.name.rsplit(".", 1)[0].replace("/", "_").replace("\\", "_")
                            zip_file.writestr(f"animated_{clean_name}.gif", gif_bytes)

                        batch_progress.progress((idx + 1) / float(total_valid))

                zip_buffer.seek(0)
                status_text.success("🎉 All animations rendered successfully!")
                batch_progress.empty()

                st.download_button(
                    "📦 Download All Animations (.ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name="calligraphy_animations.zip",
                    mime="application/zip",
                    use_container_width=True,
                )
            except Exception as exc:
                batch_progress.empty()
                status_text.error("❌ Rendering failed.")
                st.exception(exc)
