import io
import math
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
)

st.title("✒️ Artistic Calligraphy Writing Animator")
st.write(
    "Upload calligraphy artwork to automatically extract strokes, clean background noise, "
    "and generate realistic, smooth fluid-writing animations."
)

# ============================================================
# PREPROCESSING & SEGMENTATION
# ============================================================

def preprocess_and_clean_background(image):
    """
    Removes uneven paper textures and shadows using adaptive thresholding
    and morphological opening, keeping only clean artwork ink.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Estimate background lighting via large morphological dilation
    bg = cv2.dilate(gray, np.ones((19, 19), np.uint8))
    bg = cv2.medianBlur(bg, 21)
    
    # Division normalization to flatten lighting variations
    diff = 255 - cv2.absdiff(gray, bg)
    norm = cv2.normalize(diff, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    
    # Binarize ink using Otsu's thresholding
    _, binary = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Filter tiny isolated noise specs
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned_binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    return norm, cleaned_binary


def extract_stroke_components(binary_mask, min_area=80):
    """
    Extracts isolated stroke/letter components sorted left-to-right, top-to-bottom.
    """
    num_labels, cc_labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    components = []

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        cx = int(stats[label, cv2.CC_STAT_LEFT])
        cy = int(stats[label, cv2.CC_STAT_TOP])
        cw = int(stats[label, cv2.CC_STAT_WIDTH])
        ch = int(stats[label, cv2.CC_STAT_HEIGHT])

        comp_mask = (cc_labels == label)
        components.append({
            "id": len(components) + 1,
            "mask": comp_mask,
            "bbox": (cx, cy, cx + cw - 1, cy + ch - 1),
            "center": (float(centroids[label][0]), float(centroids[label][1])),
            "area": area,
        })

    # Natural reading/writing order sorting (Top-to-Bottom, Left-to-Right)
    components.sort(key=lambda p: (p["bbox"][1] // 30, p["bbox"][0]))
    return components


# ============================================================
# SKELETONIZATION & SMOOTH PATH GENERATION
# ============================================================

def morphological_skeleton(binary):
    binary_bytes = (binary.astype(np.uint8) * 255)
    skeleton = np.zeros_like(binary_bytes)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    current = binary_bytes.copy()

    while True:
        eroded = cv2.erode(current, element)
        opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(eroded, opened)
        skeleton = cv2.bitwise_or(skeleton, temp)
        current = eroded
        if cv2.countNonZero(current) == 0:
            break

    return skeleton > 0


def generate_ordered_path(skeleton, direction="Left -> Right"):
    ys, xs = np.where(skeleton)
    if len(xs) == 0:
        return []

    points = list(zip(ys.astype(int), xs.astype(int)))
    
    # Pick natural starting endpoint
    if direction == "Top -> Bottom":
        start = min(points, key=lambda p: p[0])
    elif direction == "Left -> Right":
        start = min(points, key=lambda p: p[1])
    elif direction == "Right -> Left":
        start = max(points, key=lambda p: p[1])
    else:
        start = max(points, key=lambda p: p[0])

    # Greedy Nearest-Neighbor Path Traversal
    remaining = set(points)
    remaining.remove(start)
    path = [start]
    current = start

    while remaining:
        cy, cx = current
        # Find closest point
        nearest = min(remaining, key=lambda p: ((p[0] - cy) ** 2 + (p[1] - cx) ** 2))
        
        # Avoid huge jumps (break up unconnected strokes)
        dist_sq = (nearest[0] - cy) ** 2 + (nearest[1] - cx) ** 2
        if dist_sq > 2500:  # Threshold gap
            break

        path.append(nearest)
        remaining.remove(nearest)
        current = nearest

    return path


def smooth_stroke_path(path, window_size=9):
    if len(path) < window_size:
        return path

    pts = np.array([[p[1], p[0]] for p in path], dtype=np.float32)
    kernel = np.ones(window_size, dtype=np.float32) / float(window_size)
    
    smooth_x = np.convolve(pts[:, 0], kernel, mode="same")
    smooth_y = np.convolve(pts[:, 1], kernel, mode="same")

    half = window_size // 2
    smooth_x[:half], smooth_x[-half:] = pts[:half, 0], pts[-half:, 0]
    smooth_y[:half], smooth_y[-half:] = pts[:half, 1], pts[-half:, 1]

    return [(int(round(py)), int(round(px))) for px, py in zip(smooth_x, smooth_y)]


def prepare_stroke_progress_map(component, direction):
    x1, y1, x2, y2 = component["bbox"]
    local_mask = component["mask"][y1:y2 + 1, x1:x2 + 1]
    skel = morphological_skeleton(local_mask)

    local_path = generate_ordered_path(skel, direction)
    local_path = smooth_stroke_path(local_path)
    global_path = [(y + y1, x + x1) for y, x in local_path]

    h, w = component["mask"].shape
    ys, xs = np.where(component["mask"])

    if len(xs) == 0 or not global_path:
        progress_map = np.zeros((h, w), dtype=np.float32)
        progress_map[component["mask"]] = 1.0
        return global_path, progress_map

    path_arr = np.array([[p[1], p[0]] for p in global_path], dtype=np.float32)
    pixel_points = np.column_stack((xs, ys)).astype(np.float32)

    progress_map = np.zeros((h, w), dtype=np.float32)
    distances = np.sqrt(((pixel_points[:, None, :] - path_arr[None, :, :]) ** 2).sum(axis=2))
    nearest_indices = np.argmin(distances, axis=1)

    progress_map[ys, xs] = nearest_indices.astype(np.float32) / max(1, len(path_arr) - 1)
    return global_path, progress_map


# ============================================================
# RENDERING & CALLIGRAPHY EFFECTS
# ============================================================

def ease_in_out(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def render_artistic_nib(canvas, point, angle=45, nib_size=6, color=(20, 20, 20)):
    """
    Renders an angled calligraphic pen nib at the active stroke position.
    """
    if point is None:
        return
    
    y, x = point
    rad = math.radians(angle)
    dx = int(round((nib_size / 2.0) * math.cos(rad)))
    dy = int(round((nib_size / 2.0) * math.sin(rad)))

    cv2.line(canvas, (x - dx, y - dy), (x + dx, y + dy), color, 2, lineType=cv2.LINE_AA)


def render_frame(original_img, clean_bg, components, global_progress, show_nib=True):
    canvas = clean_bg.copy()
    num_comp = len(components)
    
    active_nib_point = None

    for idx, comp in enumerate(components):
        # Stagger component start and end times
        start_t = idx / float(num_comp)
        end_t = (idx + 1) / float(num_comp)

        if global_progress < start_t:
            continue

        local_progress = (global_progress - start_t) / (end_t - start_t) if end_t > start_t else 1.0
        local_progress = ease_in_out(local_progress)

        mask = comp["mask"]
        progress_map = comp["progress_map"]
        
        # Soft-feathered stroke reveal
        reveal_threshold = local_progress + 0.02
        active_pixels = mask & (progress_map <= reveal_threshold)
        canvas[active_pixels] = original_img[active_pixels]

        if show_nib and 0.0 < local_progress < 1.0:
            path = comp.get("path", [])
            if path:
                p_idx = max(0, min(int(local_progress * (len(path) - 1)), len(path) - 1))
                active_nib_point = path[p_idx]

    if show_nib and active_nib_point is not None:
        render_artistic_nib(canvas, active_nib_point)

    return canvas


def build_gif(frames, duration=50):
    if not frames:
        return b""
    prepared = [f.convert("RGB").convert("P", palette=Image.ADAPTIVE, colors=256) for f in frames]
    buf = io.BytesIO()
    prepared[0].save(
        buf, format="GIF", save_all=True, append_images=prepared[1:],
        duration=duration, loop=0, optimize=False
    )
    return buf.getvalue()


# ============================================================
# STREAMLIT INTERFACE
# ============================================================

uploaded_file = st.file_uploader("Upload Calligraphy Image", type=["jpg", "jpeg", "png", "webp"])

st.sidebar.header("🎨 Writing Settings")
canvas_style = st.sidebar.selectbox("Background Style", ["Pure White Canvas", "Cleaned Paper Texture", "Original Image"])
writing_direction = st.sidebar.selectbox("Default Writing Direction", ["Left -> Right", "Top -> Bottom", "Right -> Left"])
show_nib = st.sidebar.checkbox("Show Calligraphy Pen Tip", value=True)
total_frames = st.sidebar.slider("Total Frames", 30, 180, 75, step=5)
gif_speed = st.sidebar.slider("Frame Delay (ms)", 20, 100, 40, step=5)

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is not None:
        norm_img, binary_mask = preprocess_and_clean_background(image)
        components = extract_stroke_components(binary_mask)

        for comp in components:
            path, p_map = prepare_stroke_progress_map(comp, writing_direction)
            comp["path"] = path
            comp["progress_map"] = p_map

        if canvas_style == "Pure White Canvas":
            base_bg = np.full_like(image, 255, dtype=np.uint8)
        elif canvas_style == "Cleaned Paper Texture":
            base_bg = cv2.cvtColor(norm_img, cv2.COLOR_GRAY2BGR)
        else:
            base_bg = image.copy()

        st.subheader("Preview & Generation")
        col1, col2 = st.columns(2)

        with col1:
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_container_width=True)

        with col2:
            st.image(binary_mask, caption=f"Extracted Clean Strokes ({len(components)} stroke components)", use_container_width=True)

        if st.button("✨ Generate Calligraphy Animation", type="primary"):
            progress_bar = st.progress(0)
            frames = []

            for f_idx in range(total_frames):
                g_progress = f_idx / float(max(1, total_frames - 1))
                frame = render_frame(image, base_bg, components, g_progress, show_nib=show_nib)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb_frame))
                progress_bar.progress((f_idx + 1) / total_frames)

            gif_data = build_gif(frames, duration=gif_speed)
            st.image(gif_data, caption="Artistic Writing Animation", use_container_width=True)

            st.download_button(
                "📥 Download GIF Animation",
                data=gif_data,
                file_name="artistic_calligraphy.gif",
                mime="image/gif"
            )
