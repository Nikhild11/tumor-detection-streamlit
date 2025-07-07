import os
# turn off Streamlit’s automatic file‑watching (so it never introspects torch.classes)
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"
import streamlit as st
from pathlib import Path
from decouple import Config, RepositoryEnv
from ultralytics import YOLO, SAM
import tempfile
import numpy as np
from PIL import Image, ImageDraw

# ─── Config ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
env = Config(RepositoryEnv(BASE_DIR / "config.env"))
MODEL_YOLO = env("TRAINED_MODEL")
MODEL_SAM = env("SAM_MODEL")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# ─── Load Models ────────────────────────────────────────────────────────────────
yolo_model = YOLO(MODEL_YOLO)
sam_model = SAM(MODEL_SAM)

# Dynamically find “No Tumor” class index
def get_no_tumor_id(model):
    for idx, name in model.names.items():
        if 'no' in name.lower() and 'tumor' in name.lower():
            return idx
    return None

NO_TUMOR_ID = get_no_tumor_id(yolo_model)

# ─── Helpers ────────────────────────────────────────────────────────────────────
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Overlay a semi-transparent mask on the base image
def overlay_mask(base_img: Image.Image, mask: np.ndarray, bbox: list, color=(0, 0, 255, 100)) -> Image.Image:
    # base_img: PIL RGB
    # mask: 2D numpy array of 0/1 within image coords
    overlay = Image.new('RGBA', base_img.size)
    draw = ImageDraw.Draw(overlay)
    # apply mask as colored region
    # iterate pixels in bbox
    x1, y1, x2, y2 = map(int, bbox)
    submask = mask[y1:y2, x1:x2]
    for yy in range(submask.shape[0]):
        for xx in range(submask.shape[1]):
            if submask[yy, xx]:
                draw.point((x1+xx, y1+yy), fill=color)
    return Image.alpha_composite(base_img.convert('RGBA'), overlay)

# ─── Streamlit App ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Brain Tumor Detection", layout="centered")
st.title("🧠 Brain Tumor Detection")

uploaded_file = st.file_uploader("Upload an MRI image", type=list(ALLOWED_EXTENSIONS))
if uploaded_file:
    if not allowed_file(uploaded_file.name):
        st.error("Invalid file type.")
    else:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        # 1) YOLO detection
        st.info("Running tumor detection...")
        det_results = yolo_model(tmp_path, save=False, verbose=False)
        det = det_results[0]

        # Annotated detection image
        annotated = det.plot()  # numpy array HWC BGR
        annotated_rgb = Image.fromarray(annotated[:, :, ::-1])  # convert BGR->RGB
        st.image(annotated_rgb, caption="Detection Result", use_container_width=True)

        # Decide message
        class_ids = det.boxes.cls.int().tolist()
        if not class_ids or (NO_TUMOR_ID is not None and all(cid == NO_TUMOR_ID for cid in class_ids)):
            st.success("No tumor is detected.")
        else:
            st.error("Tumor is detected.")

            # 2) SAM segmentation
            st.info("Running segmentation...")
            bboxes = det.boxes.xyxy.cpu().numpy().tolist()
            seg_results = sam_model(det.orig_img, bboxes=bboxes, save=False, verbose=False)

            # Overlay all masks on the detection image
            final_img = annotated_rgb.copy()
            for mask_obj, bbox in zip(seg_results[0].masks.data, bboxes):
                mask_np = mask_obj.cpu().numpy().astype(bool)
                final_img = overlay_mask(final_img, mask_np, bbox)

            # Display the overlaid result
            st.image(final_img, caption="Detection + Segmentation", use_container_width=True)

        # Clean up temp file
        os.remove(tmp_path)
