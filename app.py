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
    "extract ink strokes, and create a smooth writing animation."
)


# ============================================================
# CONSTANTS
# ============================================================

MAX_IMAGE_DIM = 600
MIN_COMPONENT_AREA = 35
MAX_SKELETON_POINTS = 12000


# ============================================================
# IMAGE UTILITIES
# ============================================================

def resize_if_large(img, max_dim=MAX_IMAGE_DIM):
    """
    Resize very large images while preserving aspect ratio.
    This keeps Streamlit Cloud memory usage under control.
    """

    if img is None:
        return None

    h, w = img.shape[:2]

    if max(h, w) <= max_dim:
        return img

    scale = max_dim / float(max(h, w))

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    return cv2.resize(
        img,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA,
    )


def decode_image(file_bytes):
    """
    Decode uploaded image bytes safely.
    """

    try:
        array = np.frombuffer(file_bytes, dtype=np.uint8)

        image = cv2.imdecode(
            array,
            cv2.IMREAD_COLOR,
        )

        return image

    except Exception:
        return None


# ============================================================
# BACKGROUND CLEANING
# ============================================================

def preprocess_and_clean_background(image):
    """
    Attempts to separate dark/colored ink from a light background.

    Returns:
        normalized grayscale image
        cleaned binary ink mask
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    # Estimate local background.
    bg_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (15, 15),
    )

    bg = cv2.dilate(
        gray,
        bg_kernel,
    )

    bg = cv2.medianBlur(
        bg,
        15,
    )

    # Difference from estimated background.
    diff = cv2.absdiff(
        gray,
        bg,
    )

    # Normalize.
    norm = cv2.normalize(
        diff,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8UC1,
    )

    # Threshold.
    _, binary = cv2.threshold(
        norm,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    # Remove tiny isolated noise.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (3, 3),
    )

    cleaned_binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1,
    )

    # Small closing helps preserve handwritten strokes.
    cleaned_binary = cv2.morphologyEx(
        cleaned_binary,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1,
    )

    return norm, cleaned_binary


# ============================================================
# STROKE COMPONENT EXTRACTION
# ============================================================

def extract_stroke_components(
    binary_mask,
    min_area=MIN_COMPONENT_AREA,
):
    """
    Extract connected ink regions.
    """

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary_mask,
            connectivity=8,
        )
    )

    components = []

    for label in range(1, num_labels):

        area = int(
            stats[label, cv2.CC_STAT_AREA]
        )

        if area < min_area:
            continue

        x = int(
            stats[label, cv2.CC_STAT_LEFT]
        )

        y = int(
            stats[label, cv2.CC_STAT_TOP]
        )

        width = int(
            stats[label, cv2.CC_STAT_WIDTH]
        )

        height = int(
            stats[label, cv2.CC_STAT_HEIGHT]
        )

        component_mask = labels == label

        components.append(
            {
                "id": len(components) + 1,
                "mask": component_mask,
                "bbox": (
                    x,
                    y,
                    x + width - 1,
                    y + height - 1,
                ),
                "center": (
                    float(centroids[label][0]),
                    float(centroids[label][1]),
                ),
                "area": area,
            }
        )

    # Reading order:
    # primarily top-to-bottom,
    # secondarily left-to-right.
    components.sort(
        key=lambda c: (
            c["bbox"][1] // 20,
            c["bbox"][0],
        )
    )

    # Re-number after sorting.
    for i, component in enumerate(components, start=1):
        component["id"] = i

    return components


# ============================================================
# FAST MORPHOLOGICAL SKELETONIZATION
# ============================================================

def morphological_skeleton(binary):
    """
    Morphological skeletonization.

    Includes a safety limit to prevent a pathological image
    from consuming excessive CPU.
    """

    binary_bytes = (
        binary.astype(np.uint8) * 255
    )

    skeleton = np.zeros_like(
        binary_bytes
    )

    element = cv2.getStructuringElement(
        cv2.MORPH_CROSS,
        (3, 3),
    )

    current = binary_bytes.copy()

    iterations = 0

    while True:

        eroded = cv2.erode(
            current,
            element,
        )

        opened = cv2.morphologyEx(
            eroded,
            cv2.MORPH_OPEN,
            element,
        )

        temp = cv2.subtract(
            eroded,
            opened,
        )

        skeleton = cv2.bitwise_or(
            skeleton,
            temp,
        )

        current = eroded

        iterations += 1

        if cv2.countNonZero(current) == 0:
            break

        # Safety protection.
        if iterations > 1000:
            break

    return skeleton > 0


# ============================================================
# PATH GENERATION
# ============================================================

def generate_ordered_path(
    skeleton,
    direction="Left -> Right",
):
    """
    Generate a reasonable writing path from a skeleton.

    This is intentionally lightweight so that it remains
    usable on Streamlit Cloud.
    """

    ys, xs = np.where(skeleton)

    if len(xs) == 0:
        return []

    # If skeleton is extremely dense, downsample points.
    if len(xs) > MAX_SKELETON_POINTS:

        step = int(
            math.ceil(
                len(xs) /
                float(MAX_SKELETON_POINTS)
            )
        )

        ys = ys[::step]
        xs = xs[::step]

    points = list(
        zip(
            ys.astype(int),
            xs.astype(int),
        )
    )

    if not points:
        return []

    # Choose starting point.
    if direction == "Top -> Bottom":

        start = min(
            points,
            key=lambda p: (
                p[0],
                p[1],
            ),
        )

    elif direction == "Right -> Left":

        start = max(
            points,
            key=lambda p: (
                p[1],
                -p[0],
            ),
        )

    else:

        start = min(
            points,
            key=lambda p: (
                p[1],
                p[0],
            ),
        )

    remaining = set(points)

    if start in remaining:
        remaining.remove(start)

    path = [start]

    current = start

    # Limit path length.
    max_path_length = min(
        len(points),
        MAX_SKELETON_POINTS,
    )

    while remaining and len(path) < max_path_length:

        cy, cx = current

        nearest = min(
            remaining,
            key=lambda p: (
                (p[0] - cy) ** 2
                +
                (p[1] - cx) ** 2
            ),
        )

        dist_sq = (
            (nearest[0] - cy) ** 2
            +
            (nearest[1] - cx) ** 2
        )

        # Allow reasonably small gaps.
        if dist_sq > 1600:
            break

        path.append(nearest)

        remaining.remove(nearest)

        current = nearest

    return path


# ============================================================
# PATH SMOOTHING
# ============================================================

def smooth_stroke_path(
    path,
    window_size=7,
):
    """
    Smooth the writing path while preserving endpoints.
    """

    if len(path) < 3:
        return path

    # Don't use a window larger than the path.
    if window_size >= len(path):
        window_size = (
            len(path)
            if len(path) % 2 == 1
            else len(path) - 1
        )

    if window_size < 3:
        return path

    pts = np.array(
        [
            [p[1], p[0]]
            for p in path
        ],
        dtype=np.float32,
    )

    kernel = (
        np.ones(
            window_size,
            dtype=np.float32,
        )
        /
        float(window_size)
    )

    smooth_x = np.convolve(
        pts[:, 0],
        kernel,
        mode="same",
    )

    smooth_y = np.convolve(
        pts[:, 1],
        kernel,
        mode="same",
    )

    half = window_size // 2

    smooth_x[:half] = pts[
        :half,
        0
    ]

    smooth_x[-half:] = pts[
        -half:,
        0
    ]

    smooth_y[:half] = pts[
        :half,
        1
    ]

    smooth_y[-half:] = pts[
        -half:,
        1
    ]

    return [
        (
            int(round(y)),
            int(round(x)),
        )
        for x, y in zip(
            smooth_x,
            smooth_y,
        )
    ]


# ============================================================
# FAST PROGRESS MAP
# ============================================================

def prepare_stroke_progress_map(
    component,
    direction,
):
    """
    Creates a map telling the renderer when each ink pixel
    should appear.

    IMPORTANT:
    The old implementation calculated the distance between
    EVERY path point and EVERY ink pixel.

    This implementation instead uses OpenCV's distance
    transform with pixel labels, which is dramatically faster.
    """

    x1, y1, x2, y2 = component["bbox"]

    local_mask = component[
        "mask"
    ][
        y1:y2 + 1,
        x1:x2 + 1
    ]

    # Skeleton.
    skeleton = morphological_skeleton(
        local_mask
    )

    # Path.
    local_path = generate_ordered_path(
        skeleton,
        direction,
    )

    local_path = smooth_stroke_path(
        local_path
    )

    global_path = [
        (
            y + y1,
            x + x1,
        )
        for y, x in local_path
    ]

    full_h, full_w = component[
        "mask"
    ].shape

    progress_map = np.zeros(
        (full_h, full_w),
        dtype=np.float32,
    )

    # No path.
    if not global_path:

        progress_map[
            component["mask"]
        ] = 1.0

        return (
            global_path,
            progress_map,
        )

    # Single point.
    if len(global_path) == 1:

        progress_map[
            component["mask"]
        ] = 1.0

        return (
            global_path,
            progress_map,
        )

    local_h, local_w = (
        local_mask.shape
    )

    # --------------------------------------------------------
    # Create an image where ONLY skeleton/path pixels are zero.
    # Distance transform finds nearest path pixel.
    # --------------------------------------------------------

    path_mask = np.ones(
        (local_h, local_w),
        dtype=np.uint8,
    )

    for py, px in local_path:

        ly = py - y1
        lx = px - x1

        if (
            0 <= ly < local_h
            and
            0 <= lx < local_w
        ):
            path_mask[
                ly,
                lx
            ] = 0

    # OpenCV labels nearest zero pixel.
    distances, labels = (
        cv2.distanceTransformWithLabels(
            path_mask,
            cv2.DIST_L2,
            5,
            labelType=cv2.DIST_LABEL_PIXEL,
        )
    )

    del distances
    del path_mask

    # --------------------------------------------------------
    # Build mapping:
    #
    # OpenCV label -> path index
    # --------------------------------------------------------

    path_positions = []

    for py, px in local_path:

        ly = py - y1
        lx = px - x1

        if (
            0 <= ly < local_h
            and
            0 <= lx < local_w
        ):
            path_positions.append(
                (
                    ly,
                    lx,
                )
            )

    label_to_progress = {}

    for index, (
        ly,
        lx,
    ) in enumerate(path_positions):

        label = int(
            labels[ly, lx]
        )

        if label > 0:
            label_to_progress[
                label
            ] = index

    # --------------------------------------------------------
    # Convert labels to progress.
    # --------------------------------------------------------

    local_component = local_mask

    component_labels = labels[
        local_component
    ]

    if component_labels.size:

        values = np.zeros(
            component_labels.shape,
            dtype=np.float32,
        )

        path_len = float(
            max(
                1,
                len(path_positions) - 1,
            )
        )

        unique_labels = np.unique(
            component_labels
        )

        for label in unique_labels:

            label_int = int(label)

            progress_index = (
                label_to_progress.get(
                    label_int,
                    len(path_positions) - 1,
                )
            )

            values[
                component_labels == label_int
            ] = (
                progress_index /
                path_len
            )

        progress_map[
            y1:y2 + 1,
            x1:x2 + 1
        ][
            local_component
        ] = values

    return (
        global_path,
        progress_map,
    )


# ============================================================
# ANIMATION FUNCTIONS
# ============================================================

def ease_in_out(t):
    """
    Smooth animation easing.
    """

    t = max(
        0.0,
        min(1.0, float(t)),
    )

    return (
        t * t *
        (3.0 - 2.0 * t)
    )


def render_artistic_nib(
    canvas,
    point,
    angle=45,
    nib_size=7,
    color=(20, 20, 20),
):
    """
    Draw a small calligraphy nib.
    """

    if point is None:
        return

    y, x = point

    rad = math.radians(
        angle
    )

    dx = int(
        round(
            (nib_size / 2.0)
            * math.cos(rad)
        )
    )

    dy = int(
        round(
            (nib_size / 2.0)
            * math.sin(rad)
        )
    )

    cv2.line(
        canvas,
        (
            x - dx,
            y - dy,
        ),
        (
            x + dx,
            y + dy,
        ),
        color,
        2,
        lineType=cv2.LINE_AA,
    )


def render_frame(
    original_img,
    clean_bg,
    components,
    global_progress,
    show_nib=True,
):
    """
    Render one animation frame.
    """

    canvas = clean_bg.copy()

    num_components = len(
        components
    )

    if num_components == 0:
        return canvas

    active_nib_point = None

    for idx, component in enumerate(
        components
    ):

        start_t = (
            idx /
            float(num_components)
        )

        end_t = (
            (idx + 1) /
            float(num_components)
        )

        if global_progress < start_t:
            continue

        if end_t > start_t:

            local_progress = (
                global_progress - start_t
            ) / (
                end_t - start_t
            )

        else:

            local_progress = 1.0

        local_progress = ease_in_out(
            local_progress
        )

        mask = component[
            "mask"
        ]

        progress_map = component[
            "progress_map"
        ]

        # Slight overlap prevents tiny gaps
        # between neighboring animation stages.
        reveal_threshold = (
            local_progress + 0.025
        )

        active_pixels = (
            mask
            &
            (
                progress_map
                <= reveal_threshold
            )
        )

        canvas[
            active_pixels
        ] = original_img[
            active_pixels
        ]

        # Current writing position.
        if (
            show_nib
            and
            0.0 < local_progress < 1.0
        ):

            path = component.get(
                "path",
                [],
            )

            if path:

                p_idx = int(
                    round(
                        local_progress
                        *
                        (
                            len(path) - 1
                        )
                    )
                )

                p_idx = max(
                    0,
                    min(
                        p_idx,
                        len(path) - 1,
                    ),
                )

                active_nib_point = (
                    path[p_idx]
                )

    if (
        show_nib
        and
        active_nib_point is not None
    ):

        render_artistic_nib(
            canvas,
            active_nib_point,
        )

    return canvas


# ============================================================
# GIF CREATION
# ============================================================

def build_gif(
    frames,
    duration=50,
):
    """
    Convert OpenCV frames to a GIF.
    """

    if not frames:
        return b""

    prepared = []

    try:

        for frame in frames:

            if isinstance(
                frame,
                Image.Image,
            ):

                pil_frame = frame.convert(
                    "RGB"
                )

            else:

                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )

                pil_frame = Image.fromarray(
                    rgb
                )

            # Palette conversion reduces GIF size.
            pil_frame = pil_frame.convert(
                "P",
                palette=Image.ADAPTIVE,
                colors=256,
            )

            prepared.append(
                pil_frame
            )

        if not prepared:
            return b""

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


# ============================================================
# IMAGE PROCESSING PIPELINE
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=20,
)
def process_uploaded_image(
    file_bytes,
    writing_direction,
    canvas_style,
):
    """
    Cached image-processing pipeline.

    This means Streamlit doesn't redo the expensive
    preprocessing every time the page reruns.
    """

    image = decode_image(
        file_bytes
    )

    if image is None:
        return None

    image = resize_if_large(
        image,
        MAX_IMAGE_DIM,
    )

    if image is None:
        return None

    norm_img, binary_mask = (
        preprocess_and_clean_background(
            image
        )
    )

    components = (
        extract_stroke_components(
            binary_mask,
            MIN_COMPONENT_AREA,
        )
    )

    if not components:
        return {
            "image": image,
            "norm_img": norm_img,
            "binary_mask": binary_mask,
            "components": [],
            "base_bg": None,
        }

    # Generate writing paths.
    for component in components:

        path, progress_map = (
            prepare_stroke_progress_map(
                component,
                writing_direction,
            )
        )

        component["path"] = path

        component[
            "progress_map"
        ] = progress_map

    # --------------------------------------------------------
    # Background
    # --------------------------------------------------------

    if (
        canvas_style
        ==
        "Pure White Canvas"
    ):

        base_bg = np.full_like(
            image,
            255,
            dtype=np.uint8,
        )

    elif (
        canvas_style
        ==
        "Cleaned Paper Texture"
    ):

        base_bg = cv2.cvtColor(
            norm_img,
            cv2.COLOR_GRAY2BGR,
        )

        # Make the cleaned paper less extreme.
        base_bg = cv2.normalize(
            base_bg,
            None,
            215,
            255,
            cv2.NORM_MINMAX,
        )

    else:

        base_bg = image.copy()

    return {
        "image": image,
        "norm_img": norm_img,
        "binary_mask": binary_mask,
        "components": components,
        "base_bg": base_bg,
    }


# ============================================================
# STREAMLIT SIDEBAR
# ============================================================

st.sidebar.header(
    "🎨 Global Writing Settings"
)

canvas_style = st.sidebar.selectbox(
    "Background Style",
    [
        "Pure White Canvas",
        "Cleaned Paper Texture",
        "Original Image",
    ],
)

writing_direction = st.sidebar.selectbox(
    "Default Writing Direction",
    [
        "Left -> Right",
        "Top -> Bottom",
        "Right -> Left",
    ],
)

show_nib = st.sidebar.checkbox(
    "Show Calligraphy Pen Tip",
    value=True,
)

total_frames = st.sidebar.slider(
    "Total Frames per GIF",
    min_value=30,
    max_value=120,
    value=60,
    step=5,
)

gif_speed = st.sidebar.slider(
    "Frame Delay (ms)",
    min_value=20,
    max_value=100,
    value=40,
    step=5,
)


# ============================================================
# FILE UPLOADER
# ============================================================

uploaded_files = st.file_uploader(
    "Upload Calligraphy Images (Multiple Selection Supported)",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp",
    ],
    accept_multiple_files=True,
)


# ============================================================
# MAIN PROCESSING
# ============================================================

if uploaded_files:

    st.info(
        f"📁 {len(uploaded_files)} file(s) uploaded. "
        "The images will be cleaned and prepared automatically."
    )

    tabs = st.tabs(
        [
            f"🖼️ {file.name}"
            for file in uploaded_files
        ]
    )

    processed_data = {}

    # --------------------------------------------------------
    # PROCESS EACH IMAGE
    # --------------------------------------------------------

    for tab, file in zip(
        tabs,
        uploaded_files,
    ):

        with tab:

            file.seek(0)

            file_bytes = file.read()

            if not file_bytes:

                st.error(
                    f"❌ `{file.name}` is empty."
                )

                continue

            # Cached processing.
            with st.spinner(
                f"Preparing {file.name}..."
            ):

                try:

                    result = (
                        process_uploaded_image(
                            file_bytes,
                            writing_direction,
                            canvas_style,
                        )
                    )

                except Exception as exc:

                    st.error(
                        f"❌ Could not process "
                        f"`{file.name}`."
                    )

                    st.exception(
                        exc
                    )

                    continue

            if result is None:

                st.error(
                    f"❌ Could not read `{file.name}` "
                    "as an image."
                )

                continue

            components = result[
                "components"
            ]

            if not components:

                st.warning(
                    f"⚠️ No valid ink strokes detected "
                    f"in `{file.name}`.\n\n"
                    "Try an image with stronger contrast "
                    "between the drawing and background."
                )

                continue

            # ------------------------------------------------
            # PREVIEW
            # ------------------------------------------------

            col1, col2 = st.columns(
                2
            )

            with col1:

                st.image(
                    cv2.cvtColor(
                        result["image"],
                        cv2.COLOR_BGR2RGB,
                    ),
                    caption="Uploaded Image",
                    use_container_width=True,
                )

            with col2:

                st.image(
                    result["binary_mask"],
                    caption=(
                        f"Extracted Clean Strokes "
                        f"({len(components)} components)"
                    ),
                    use_container_width=True,
                )

            # ------------------------------------------------
            # DETAILS
            # ------------------------------------------------

            total_ink_pixels = int(
                np.count_nonzero(
                    result["binary_mask"]
                )
            )

            st.caption(
                f"Detected {len(components)} stroke "
                f"components • "
                f"{total_ink_pixels:,} ink pixels"
            )

            processed_data[
                file.name
            ] = result

    # ========================================================
    # RENDER SECTION
    # ========================================================

    st.markdown("---")

    st.subheader(
        "🎬 Generate Animations"
    )

    st.write(
        "When you click the button below, each valid "
        "image will be rendered as an animated GIF "
        "and all GIFs will be placed into one ZIP file."
    )

    if st.button(
        "🚀 Render All Animations & Download ZIP",
        type="primary",
        use_container_width=True,
    ):

        if not processed_data:

            st.error(
                "❌ No valid images are available to render."
            )

        else:

            batch_progress = st.progress(
                0.0
            )

            status_text = st.empty()

            zip_buffer = io.BytesIO()

            valid_files = [
                f
                for f in uploaded_files
                if f.name in processed_data
            ]

            total_valid = len(
                valid_files
            )

            try:

                with zipfile.ZipFile(
                    zip_buffer,
                    "w",
                    compression=zipfile.ZIP_DEFLATED,
                ) as zip_file:

                    for idx, file in enumerate(
                        valid_files
                    ):

                        status_text.write(
                            f"🎬 Rendering "
                            f"**{file.name}** "
                            f"({idx + 1}/{total_valid})..."
                        )

                        item = processed_data[
                            file.name
                        ]

                        frames = []

                        # ------------------------------------
                        # Generate frames
                        # ------------------------------------

                        for frame_index in range(
                            total_frames
                        ):

                            if total_frames <= 1:

                                global_progress = 1.0

                            else:

                                global_progress = (
                                    frame_index
                                    /
                                    float(
                                        total_frames - 1
                                    )
                                )

                            frame = render_frame(
                                item["image"],
                                item["base_bg"],
                                item["components"],
                                global_progress,
                                show_nib=show_nib,
                            )

                            frames.append(
                                Image.fromarray(
                                    cv2.cvtColor(
                                        frame,
                                        cv2.COLOR_BGR2RGB,
                                    )
                                )

                            )

                        # ------------------------------------
                        # Build GIF
                        # ------------------------------------

                        gif_bytes = build_gif(
                            frames,
                            duration=gif_speed,
                        )

                        # Release frame references.
                        frames.clear()

                        if gif_bytes:

                            clean_name = (
                                file.name
                                .rsplit(
                                    ".",
                                    1,
                                )[0]
                            )

                            # Remove potentially problematic
                            # characters from ZIP filename.
                            clean_name = (
                                clean_name
                                .replace(
                                    "/",
                                    "_",
                                )
                                .replace(
                                    "\\",
                                    "_",
                                )
                                .replace(
                                    ":",
                                    "_",
                                )
                            )

                            zip_file.writestr(
                                f"animated_{clean_name}.gif",
                                gif_bytes,
                            )

                        else:

                            st.warning(
                                f"⚠️ GIF creation failed "
                                f"for `{file.name}`."
                            )

                        batch_progress.progress(
                            (idx + 1)
                            /
                            float(total_valid)
                        )

                zip_buffer.seek(0)

                status_text.success(
                    "🎉 All animations rendered successfully!"
                )

                batch_progress.empty()

                zip_data = (
                    zip_buffer.getvalue()
                )

                st.success(
                    f"📦 ZIP ready — "
                    f"{len(zip_data) / (1024 * 1024):.2f} MB"
                )

                st.download_button(
                    "📦 Download All Animations (.ZIP)",
                    data=zip_data,
                    file_name=(
                        "calligraphy_animations.zip"
                    ),
                    mime="application/zip",
                    use_container_width=True,
                )

            except Exception as exc:

                batch_progress.empty()

                status_text.error(
                    "❌ Animation rendering failed."
                )

                st.exception(
                    exc
                )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "✒️ Artistic Calligraphy Animator • "
    "Optimized for batch processing and Streamlit Cloud"
)
