import os, io, base64, concurrent.futures
import cv2
import io
from PIL import Image, ImageCms, ImageOps, ImageFilter, ImageEnhance
import numpy as np
import piexif
import torch  # For GPU/CPU support
import multiprocessing  # For multi-core processing
import sys, cProfile, pstats
import rawpy
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QProgressBar,
    QCheckBox,
    QGridLayout,
    QHBoxLayout,
    QVBoxLayout,
    QVBoxLayout,
    QLabel,
    QComboBox,
    QSlider,
    QFormLayout,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject, QTimer
from PyQt5.QtGui import QImage, QPixmap
import torchvision.ops as ops

# ----------------------------
# Monkey-Patch NMS for CUDA fallback
# ----------------------------
_original_nms = ops.nms


def nms_cpu_fallback(boxes, scores, iou_threshold):
    boxes_cpu = boxes.cpu()
    scores_cpu = scores.cpu()
    keep = _original_nms(boxes_cpu, scores_cpu, iou_threshold)
    return keep.to(boxes.device)


ops.nms = nms_cpu_fallback
# ----------------------------
# Device Setup (GPU/CPU)
# ----------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Using CPU.")

# ----------------------------
# Face Detection Model
# ----------------------------
from retinaface.pre_trained_models import get_model

model = get_model("resnet50_2020-07-20", max_size=2048, device=device)
model.eval()

# ----------------------------
# HEIC/HEIF Support Setup
# ----------------------------
try:
    import pillow_heif

    pillow_heif.register_heif_opener()
    if hasattr(pillow_heif, "register_heif_saver"):
        pillow_heif.register_heif_saver()
    else:
        print("Warning: HEIC saver not available in this pillow-heif version.")
except ImportError:
    print("pillow-heif not installed; HEIC support will be limited.")
except Exception as e:
    print("Error setting up HEIC support:", e)


# ----------------------------
# Core Functionality
# ----------------------------
def create_required_folders():
    REQUIRED_FOLDERS = ["originals", "face_detector", "cropped"]
    for folder in REQUIRED_FOLDERS:
        os.makedirs(folder, exist_ok=True)


# ----------------------------
# Embedded ICC Profiles
# ----------------------------
SRGB_PROFILE_BASE64 = """
AAAMYWxzdGF0aWMteHJkZi1zdHJlYW0cbjI7ADxzdHJlYW0KbWFqb3I6IDEKbWlub3I6IDAK
YmV0YTogMApjb25kaXRpb25zIDAKZW5kYXZvcjogMApvcGVyYXRvcl9uYW1lOiBQYXJzZWQg
U1JHQiBwcm9maWxlCmNvcHlyaWdodDogQ29weXJpZ2h0IEFwcGxlIEluYy4sIDE5OTkKbWFu
dWZhY3R1cmVyOiBBcHBsZQptb2RlbDogMQpzdGFydGluZ19vZmZzZXQ6IDAKc3RvcHBpbmdf
b2Zmc2V0OiAwCnNpZ25hdHVyZTogc3JnYgpkZXNjcmlwdGlvbjogU1JHQiBjb2xvciBwcm9m
aWxlCmRlc2NyaXB0aW9uX3N0cmluZzogU1JHQiBjb2xvciBwcm9maWxlCmNvbm5lY3Rpb25f
dHlwZTogUkdCCnByb2ZpbGVfaWQ6IDAKY2xvc2luZ19sYWJlbDogRW5kIG9mIHByb2ZpbGUK
ZW5kX2Jsb2NrX3NpZ25hdHVyZTogZW9jcApleGlmX3ZlcnNpb246IDIuMgpjb2xvcl9zcGFj
ZTogU1JHQgpjb21wcmVzc2lvbjogMApiaXRzX3Blcl9jb21wb25lbnQ6IDgKd2lkdGg6IDAK
aGVpZ2h0OiAwCmNvbXByZXNzaW9uX3R5cGU6IDAKcGhvdG9tZXRyaWNfaW50ZXJwcmV0YXRp
b246IDAKZGF0ZV90aW1lOiAxOTk5OjAxOjAxIDAwOjAwOjAwCnN0cmlwX29mZnNldHM6IDAK
cm93c19wZXJfc3RyaXA6IDAKc3RyaXBfYnl0ZV9jb3VudHM6IDAKcGxhbmFyX2NvbmZpZ3Vy
YXRpb246IDAKc2FtcGxlX2Zvcm1hdDogMApzbWFydF9zdHJpcF9vZmZzZXQ6IDAKcHJlZGlj
dG9yOiAwCnBhZGRpbmc6IDAKY29sb3JfbWFwX3R5cGU6IDAKY29sb3JfbWFwX2xlbmd0aDog
MApyZWRfdHlwZTogMApyZWRfY29sX3R5cGU6IDAKcmVkX2xlbmd0aDogMApncmVlbl90eXBl
OiAwCmdncmVlbl9jb2xfdHlwZTogMApncmVlbl9sZW5ndGg6IDAKYmx1ZV90eXBlOiAwCmJs
dWVfY29sX3R5cGU6IDAKYmx1ZV9sZW5ndGg6IDAKcmVkX3gfb3JpZ2luOiAwCnJlZF95X29y
aWdpbjogMApncmVlbl94X29yaWdpbjogMApncmVlbl95X29yaWdpbjogMApibHVlX3hfb3Jp
Z2luOiAwCmJsdWVfeV9vcmlnaW46IDAKcmVkX3o6IDAKcmVkX3k6IDAKZ3JlZW5feDogMApn
cmVlbl95OiAwCmJsdWVfeDogMApibHVlX2NvbG9yX3R5cGU6IDAKZ3JlZW5fY29sb3JfdHlw
ZTogMApibHVlX2NvbG9yX3R5cGU6IDAKcmVkX2NvbG9yX2xlbmd0aDogMApncmVlbl9jb2xv
cl9sZW5ndGg6IDAKYmx1ZV9jb2xvcl9sZW5ndGg6IDAKY2FsbGJhY2tfdHlwZTogMApjYWxs
YmFja19vZmZzZXQ6IDAKY2FsbGJhY2tfc2l6ZTogMApjYWxsYmFja19wYXJhbTogMApmaWxs
X29yZGVyOiAwCnVua25vd24xOiAwCnVua25vd24yOiAwCnVua25vd24zOiAwCnVua25vd240
OiAwCnVua25vd241OiAwCnVua25vd242OiAwCnVua25vd243OiAwCnVua25vd244OiAwCnVu
a25vd245OiAwCmVuZG9mZmxpbmU6IDAKZW5kb2ZmaWxlOiAwCmVuZG9mZmlsZTozMDA7AA== 
"""
SRGB_PROFILE = base64.b64decode(SRGB_PROFILE_BASE64.replace("\n", ""))

try:
    with open("Byte64.txt", "rb") as f:
        data = f.read().strip()
    try:
        base64_text = data.decode("ascii")
    except UnicodeDecodeError:
        try:
            base64_text = data.decode("utf-16")
        except UnicodeDecodeError:
            base64_text = data.decode("utf-8-sig")
    DISPLAY_P3_PROFILE = base64.b64decode(base64_text)
except Exception as e:
    print("Error reading Display P3 profile from Byte64.txt:", e)
    DISPLAY_P3_PROFILE = None

# ----------------------------
# Color Conversion Helpers
# ----------------------------
icc_transform_cache = {}


def get_icc_transform(input_icc, mode):
    if input_icc is None:
        input_icc = SRGB_PROFILE
    key = (input_icc, mode)
    if key in icc_transform_cache:
        return icc_transform_cache[key]
    try:
        in_profile = ImageCms.ImageCmsProfile(io.BytesIO(input_icc))
        out_profile = ImageCms.ImageCmsProfile(io.BytesIO(DISPLAY_P3_PROFILE))
        transform = ImageCms.buildTransformFromOpenProfiles(
            in_profile, out_profile, mode, mode
        )
        icc_transform_cache[key] = transform
        return transform
    except Exception as e:
        print(f"Error building ICC transform: {e}")
        return None


def convert_to_displayp3(pil_img, input_icc=None):
    if input_icc is None:
        input_icc = SRGB_PROFILE
    transform = get_icc_transform(input_icc, pil_img.mode)
    if transform is None:
        return pil_img
    try:
        converted_img = ImageCms.applyTransform(pil_img, transform)
        return converted_img
    except Exception as e:
        print("Error converting image to Display P3:", e)
        return pil_img


def process_color_profile(pil_img, metadata):
    input_icc = metadata.get("icc_profile")
    in_desc = "sRGB"
    if input_icc:
        try:
            with io.BytesIO(input_icc) as f:
                in_profile = ImageCms.ImageCmsProfile(f)
                in_desc = ImageCms.getProfileDescription(in_profile)
        except Exception as e:
            print(f"Profile read error: {e}")
            in_desc = "sRGB"
    if "Display P3" not in in_desc:
        print(f"Converting from {in_desc} to Display P3")
        transform = get_icc_transform(input_icc, pil_img.mode)
        if transform:
            try:
                converted = ImageCms.applyTransform(pil_img, transform)
                converted.info["icc_profile"] = DISPLAY_P3_PROFILE
                return converted
            except Exception as e:
                print(f"Profile conversion failed: {e}")
    else:
        print("Image already in Display P3")
    pil_img.info["icc_profile"] = DISPLAY_P3_PROFILE
    return pil_img


# ----------------------------
# Image I/O, Face Detection, and Cropping
# ----------------------------
def save_as_heic_fallback(cropped_img, output_path):
    try:
        if cropped_img.mode != "RGB":
            cropped_img = cropped_img.convert("RGB")
        heif_file = pillow_heif.HeifFile()
        heif_file.add_image(
            cropped_img.tobytes(), width=cropped_img.width, height=cropped_img.height
        )
        heif_file.save(output_path, quality=100)
        return True
    except Exception as e:
        print(f"HEIC save failed: {e}")
        return False


try:
    import rawpy
except ImportError:
    print("rawpy not installed; RAW support will not be available.")

def read_image(input_path, max_dim=1024, sharpen=True, enhance_lighting=False):
    """
    Read an image from a file path, with optional resizing, sharpening, and lighting enhancement.
    Supports RAW, HEIC, and standard image formats.

    Args:
        input_path (str): Path to the image file
        max_dim (int): Maximum dimension for resizing
        sharpen (bool): Apply sharpening filter after resizing
        enhance_lighting (bool): Apply lighting enhancement for improved face detection

    Returns:
        tuple: (OpenCV image, PIL image, metadata dictionary)
    """
    # --- RAW File Handling ---
    raw_extensions = ('.cr2', '.nef', '.arw', '.dng', '.orf', '.raf')
    if input_path.lower().endswith(raw_extensions):
        try:
            with rawpy.imread(input_path) as raw:
                rgb = raw.postprocess()
            pil_img = Image.fromarray(rgb)
            scale = min(max_dim / pil_img.width, max_dim / pil_img.height, 1)
            if scale < 1:
                new_size = (int(pil_img.width * scale), int(pil_img.height * scale))
                pil_img = pil_img.resize(new_size, Image.LANCZOS)
                if sharpen:
                    pil_img = pil_img.filter(ImageFilter.SHARPEN)
            metadata = pil_img.info.copy()
            pil_img = pil_img.convert("RGB")
            cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            if enhance_lighting:
                cv_img = enhance_lighting_for_faces(cv_img)
                pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
            return cv_img, pil_img, metadata
        except Exception as e:
            print(f"RAW image read error: {e}")
            return None, None, {}

    # --- HEIC/HEIF Handling ---
    if input_path.lower().endswith(".heic"):
        try:
            heif_file = pillow_heif.read_heif(input_path)
            pil_img = Image.frombytes(
                heif_file.mode, heif_file.size, heif_file.data, "raw"
            )
            metadata = {}
            try:
                if hasattr(heif_file, "color_profile"):
                    metadata["icc_profile"] = heif_file.color_profile["data"]
                elif (
                    hasattr(heif_file, "metadata")
                    and "icc_profile" in heif_file.metadata
                ):
                    metadata["icc_profile"] = heif_file.metadata["icc_profile"]
                elif hasattr(heif_file, "info") and "icc_profile" in heif_file.info:
                    metadata["icc_profile"] = heif_file.info["icc_profile"]
            except Exception as e:
                print(f"Color profile extraction warning: {e}")
            if hasattr(heif_file, "metadata"):
                metadata.update(heif_file.metadata)
            cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            # Apply lighting enhancement if requested
            if enhance_lighting:
                cv_img = enhance_lighting_for_faces(cv_img)
                # Update PIL image to match enhanced OpenCV image
                pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

            return cv_img, pil_img, metadata
        except Exception as e:
            print(f"HEIC read error: {e}")
            return None, None, {}

    # --- Standard Image Handling ---
    else:
        try:
            pil_img = ImageOps.exif_transpose(Image.open(input_path))
            scale = min(max_dim / pil_img.width, max_dim / pil_img.height, 1)
            if scale < 1:
                new_size = (int(pil_img.width * scale), int(pil_img.height * scale))
                pil_img = pil_img.resize(new_size, Image.LANCZOS)
                if sharpen:
                    pil_img = pil_img.filter(ImageFilter.SHARPEN)
            metadata = pil_img.info.copy()
            try:
                metadata["exif"] = piexif.dump(piexif.load(input_path))
            except Exception:
                pass
            pil_img = pil_img.convert("RGB")
            cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            # Apply lighting enhancement if requested
            if enhance_lighting:
                cv_img = enhance_lighting_for_faces(cv_img)
                # Update PIL image to match enhanced OpenCV image
                pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

            return cv_img, pil_img, metadata
        except Exception as e:
            print(f"Image read error: {e}")
            return None, None, {}


def enhance_lighting_for_faces(cv_img):
    """
    Enhance image lighting to improve face detection in challenging conditions
    """
    # Convert to LAB color space (L=lightness, A=green-red, B=blue-yellow)
    lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l)

    enhanced_lab = cv2.merge((enhanced_l, a, b))

    # Convert back to BGR color space
    enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    hsv = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Only brighten pixels below certain brightness threshold
    mask = v < 100
    v[mask] = np.clip(v[mask] * 1.3, 0, 255).astype(np.uint8)

    enhanced_hsv = cv2.merge((h, s, v))
    result = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)

    return result


def correct_rotation(cv_img, landmarks):
    """Correct rotation based on eye positions"""
    left_eye = np.array(landmarks["left_eye"])
    right_eye = np.array(landmarks["right_eye"])

    # Calculate angle for rotation
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))

    # Only correct if rotation is significant (e.g., > 5 degrees)
    if abs(angle) < 5:
        return cv_img, landmarks

    # Get image dimensions
    height, width = cv_img.shape[:2]
    center = (width // 2, height // 2)

    # Calculate rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_img = cv2.warpAffine(
        cv_img, rotation_matrix, (width, height), borderMode=cv2.BORDER_REPLICATE
    )

    # Update landmarks by applying the same rotation
    updated_landmarks = {}
    for key, point in landmarks.items():
        # Create homogeneous coordinate
        point_array = np.array([point[0], point[1], 1])
        # Apply rotation
        new_point = np.dot(rotation_matrix, point_array)
        updated_landmarks[key] = (new_point[0], new_point[1])

    return rotated_img, updated_landmarks


def get_face_and_landmarks(
    input_path, conf_threshold=0.3, sharpen=True, apply_rotation=True
):
    cv_img, pil_img, metadata = read_image(input_path, sharpen=sharpen)
    if cv_img is None:
        return None, None, None, None, metadata

    try:
        with torch.no_grad():
            annotations = model.predict_jsons(cv_img)
    except Exception as e:
        print(f"Detection error: {e}")
        return None, None, None, None, metadata

    valid_detections = [det for det in annotations if det["score"] >= conf_threshold]
    if not valid_detections:
        return None, None, None, None, metadata

    best_det = max(valid_detections, key=lambda x: x["score"])
    if "landmarks" not in best_det or len(best_det["landmarks"]) < 5:
        return None, None, None, None, metadata

    box = best_det.get("bbox")
    landmarks = {
        "left_eye": best_det["landmarks"][0],
        "right_eye": best_det["landmarks"][1],
        "nose": best_det["landmarks"][2],
        "mouth_left": best_det["landmarks"][3],
        "mouth_right": best_det["landmarks"][4],
    }

    if apply_rotation:
        corrected_cv_img, corrected_landmarks = correct_rotation(cv_img, landmarks)
        corrected_pil_img = Image.fromarray(
            cv2.cvtColor(corrected_cv_img, cv2.COLOR_BGR2RGB)
        )
        return box, corrected_landmarks, corrected_cv_img, corrected_pil_img, metadata

    return box, landmarks, cv_img, pil_img, metadata


def is_frontal_face(landmarks):
    left_eye = np.array(landmarks["left_eye"], dtype="float")
    right_eye = np.array(landmarks["right_eye"], dtype="float")
    nose = np.array(landmarks["nose"], dtype="float")
    d_left = np.linalg.norm(nose - left_eye)
    d_right = np.linalg.norm(nose - right_eye)
    ratio = min(d_left, d_right) / max(d_left, d_right)
    diff = abs(d_left - d_right)
    avg = (d_left + d_right) / 2.0
    rel_diff = diff / avg
    print(
        f"Eye-to-nose ratio: {ratio:.2f}, absolute diff: {diff:.2f}, relative diff: {rel_diff:.2f}"
    )
    return (ratio >= 0.70) and (rel_diff <= 0.22)


def save_image(cropped_img, output_path, metadata):
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        file_ext = os.path.splitext(output_path)[1].lower()
        
        if file_ext == ".heic":
            if cropped_img.mode != "RGB":
                cropped_img = cropped_img.convert("RGB")
            cropped_img.save(
                output_path,
                format="HEIF",
                quality=100,
                save_all=True,
                matrix_coefficients=0,
                chroma=444,
                icc_profile=DISPLAY_P3_PROFILE,
            )
            return True
        
        elif file_ext in (".tiff", ".tif"):
            if cropped_img.mode != "RGB":
                cropped_img = cropped_img.convert("RGB")
            cropped_img.save(
                output_path,
                format="TIFF",
                icc_profile=DISPLAY_P3_PROFILE,
                **metadata
            )
            return True
        
        elif file_ext == ".dng":
            # Attempt to use a dedicated DNG writer library (hypothetically pydng)
            try:
                import pydng
                pydng.write_dng(cropped_img, output_path, metadata=metadata, icc_profile=DISPLAY_P3_PROFILE)
                return True
            except ImportError:
                print("pydng not installed; falling back to TIFF for DNG output.")
                fallback_path = os.path.splitext(output_path)[0] + ".tiff"
                if cropped_img.mode != "RGB":
                    cropped_img = cropped_img.convert("RGB")
                cropped_img.save(
                    fallback_path,
                    format="TIFF",
                    icc_profile=DISPLAY_P3_PROFILE,
                    **metadata
                )
                return True
        
        else:
            # For all other formats, save normally using PIL's save (which will use metadata)
            cropped_img.save(output_path, **metadata)
            return True
    except Exception as e:
        print(f"Save error: {e}")
        return False



def crop_frontal_image(
    pil_img, frontal_margin=20, landmarks=None, metadata={}, lip_offset=50
):
    width, height = pil_img.size
    if not landmarks or not all(k in landmarks for k in ["mouth_left", "mouth_right"]):
        print("Lip landmarks are missing.")
        return None
    lip_y = (landmarks["mouth_left"][1] + landmarks["mouth_right"][1]) / 2
    crop_top = max(0, int(lip_y) - lip_offset)
    crop_left = frontal_margin
    crop_right = width - frontal_margin
    crop_bottom = height
    try:
        cropped_img = pil_img.crop((crop_left, crop_top, crop_right, crop_bottom))
        return process_color_profile(cropped_img, metadata)
    except Exception as e:
        print(f"Crop error: {e}")
        return None


def crop_profile_image(
    pil_img, profile_margin=20, neck_offset=50, box=None, metadata={}
):
    width, height = pil_img.size
    if box is None or len(box) < 4:
        print("Bounding box is missing or invalid.")
        return None
    crop_top = min(box[3] + neck_offset, height - 1)
    try:
        cropped_img = pil_img.crop(
            (profile_margin, crop_top, width - profile_margin, height)
        )
        return process_color_profile(cropped_img, metadata)
    except Exception as e:
        print(f"Crop error: {e}")
        return None


def auto_crop(
    pil_img,
    frontal_margin,
    profile_margin,
    box,
    landmarks,
    metadata,
    lip_offset=50,
    neck_offset=50,
):
    if is_frontal_face(landmarks):
        print("Using crop_frontal_image")
        cropped_image = crop_frontal_image(
            pil_img, frontal_margin, landmarks, metadata, lip_offset=lip_offset
        )
        if cropped_image is None:
            print("crop_frontal_image returned None")  # ADDED
        return cropped_image
    else:
        print("Using crop_profile_image")
        cropped_image = crop_profile_image(pil_img, profile_margin, neck_offset, box, metadata)
        if cropped_image is None:
            print("crop_profile_image returned None")  # ADDED
        return cropped_image


def crop_chin_image(pil_img, margin=20, box=None, metadata={}, chin_offset=20):
    if box is None or len(box) < 4:
        return None
    width, height = pil_img.size
    crop_top = max(0, box[3] - chin_offset)
    crop_left = margin
    crop_right = width - margin
    crop_bottom = height
    try:
        cropped_img = pil_img.crop((crop_left, crop_top, crop_right, crop_bottom))
        return process_color_profile(cropped_img, metadata)
    except Exception as e:
        print(f"Chin crop error: {e}")
        return None


def crop_nose_image(pil_img, box, landmarks, metadata={}, margin=0):
    x1, y1, x2, y2 = box
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(pil_img.width, x2 + margin)
    y2 = min(pil_img.height, y2 + margin)
    try:
        cropped_img = pil_img.crop((x1, y1, x2, y2))
        return process_color_profile(cropped_img, metadata)
    except Exception as e:
        print(f"Nose crop error: {e}")
        return None


def crop_below_lips_image(pil_img, margin=20, landmarks=None, metadata={}, offset=10):
    if not landmarks or not all(k in landmarks for k in ["mouth_left", "mouth_right"]):
        return None
    width, height = pil_img.size
    lip_y = (landmarks["mouth_left"][1] + landmarks["mouth_right"][1]) / 2
    crop_top = min(height, int(lip_y + offset))
    crop_left = margin
    crop_right = width - margin
    crop_bottom = height
    try:
        cropped_img = pil_img.crop((crop_left, crop_top, crop_right, crop_bottom))
        return process_color_profile(cropped_img, metadata)
    except Exception as e:
        print(f"Below lips crop error: {e}")
        return None


def crop_frontal_image_preview(
    pil_img, frontal_margin=20, landmarks=None, metadata={}, lip_offset=50
):
    width, height = pil_img.size
    print(f"Cropping frontal (preview): {width}x{height}")
    if not landmarks or not all(k in landmarks for k in ["mouth_left", "mouth_right"]):
        print("Lip landmarks are missing for preview.")
        return None
    lip_y = (landmarks["mouth_left"][1] + landmarks["mouth_right"][1]) / 2
    crop_top = max(0, int(lip_y) - lip_offset)
    crop_left = frontal_margin
    crop_right = width - frontal_margin
    crop_bottom = height
    print(f"Preview crop coordinates: {crop_left, crop_top, crop_right, crop_bottom}")
    try:
        cropped_img = pil_img.crop((crop_left, crop_top, crop_right, crop_bottom))
        cropped_img = process_color_profile(cropped_img, metadata)
        return cropped_img
    except Exception as e:
        print(f"Preview crop error: {e}")
        return None


def crop_profile_image_preview(
    pil_img, profile_margin=20, neck_offset=50, box=None, metadata={}
):
    width, height = pil_img.size
    if box is None or len(box) < 4:
        print("Bounding box is missing or invalid for preview.")
        return None
    crop_top = min(box[3] + neck_offset, height - 1)
    try:
        cropped_img = pil_img.crop(
            (profile_margin, crop_top, width - profile_margin, height)
        )
        cropped_img = process_color_profile(cropped_img, metadata)
        return cropped_img
    except Exception as e:
        print(f"Preview crop error: {e}")
        return None


def process_batch(
    batch_filenames,
    input_folder,
    output_folder,
    frontal_margin,
    profile_margin,
    sharpen=True,
    use_frontal=True,
    use_profile=True,
    apply_rotation=True,
    crop_style="frontal",
    filter_name="None",
    filter_intensity=50,
    aspect_ratio=None  # New parameter for aspect ratio (float, e.g., 16/9)
):
    count = 0
    for filename in batch_filenames:
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, f"cropped_{filename}")
        try:
            result = get_face_and_landmarks(
                input_path, sharpen=sharpen, apply_rotation=apply_rotation
            )
            if result is None or result[0] is None or result[1] is None:
                print(f"{filename}: No face detected. Skipping...")
            else:
                box, landmarks, _, pil_img, metadata = result
                crop_functions = {
                    "frontal": lambda: (
                        crop_frontal_image(
                            pil_img, frontal_margin, landmarks, metadata, lip_offset=50
                        )
                        if use_frontal and is_frontal_face(landmarks)
                        else auto_crop(
                            pil_img,
                            frontal_margin,
                            profile_margin,
                            box,
                            landmarks,
                            metadata,
                            lip_offset=50,
                            neck_offset=50,
                        )
                    ),
                    "profile": lambda: (
                        crop_profile_image(pil_img, profile_margin, 50, box, metadata)
                        if use_profile
                        else None
                    ),
                    "chin": lambda: crop_chin_image(
                        pil_img, frontal_margin, box, metadata, chin_offset=20
                    ),
                    "nose": lambda: crop_nose_image(
                        pil_img, box, landmarks, metadata, margin=0
                    ),
                    "below_lips": lambda: crop_below_lips_image(
                        pil_img, frontal_margin, landmarks, metadata, offset=10
                    ),
                    "auto": lambda: auto_crop(
                        pil_img,
                        frontal_margin,
                        profile_margin,
                        box,
                        landmarks,
                        metadata,
                        lip_offset=50,
                        neck_offset=50
                    ),
                }
                cropped_img = crop_functions.get(crop_style, lambda: None)()
                if cropped_img and aspect_ratio:
                    cropped_img = apply_aspect_ratio_filter(cropped_img, aspect_ratio)
                if cropped_img:
                    cropped_img = apply_filter(cropped_img, filter_name, filter_intensity)
                    # Save the cropped image and capture the success status
                    saved = save_image(cropped_img, output_path, metadata)
                else:
                    print(f"{filename}: Cropping failed. Skipping...")
                    saved = False
                # Remove original only if the save succeeded and paths differ
                if saved and os.path.dirname(input_path) != os.path.dirname(output_path):
                    os.remove(input_path)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
        finally:
            count += 1
    return count


def process_images_threaded(
    input_folder,
    output_folder,
    frontal_margin,
    profile_margin,
    sharpen=True,
    use_frontal=True,
    use_profile=True,
    progress_callback=None,
    cancel_func=None,
    apply_rotation=True,
    crop_style="auto",  # Changed default to "auto"
    filter_name="None",
    filter_intensity=50,
    aspect_ratio=None  # New parameter for aspect ratio
):
    os.makedirs(output_folder, exist_ok=True)
    valid_exts = (".jpg", ".jpeg", ".png", ".heic")
    filenames = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_exts)]
    total = len(filenames)
    if total == 0:
        return 0, 0
    batch_size = max(1, total // (multiprocessing.cpu_count() * 2))
    max_workers = max(1, min(4, len(filenames) // batch_size))
    batches = [filenames[i : i + batch_size] for i in range(0, total, batch_size)]
    processed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {
            executor.submit(
                process_batch,
                batch,
                input_folder,
                output_folder,
                frontal_margin,
                profile_margin,
                sharpen,
                use_frontal,
                use_profile,
                apply_rotation,
                crop_style,
                filter_name,
                filter_intensity,
                aspect_ratio  # Passing aspect ratio along
            ): batch
            for batch in batches
        }
        for future in concurrent.futures.as_completed(future_to_batch):
            if cancel_func and cancel_func():
                break
            try:
                batch_count = future.result()
                processed += batch_count
                if progress_callback:
                    progress_callback(processed, total, "Batch processed")
            except Exception as e:
                print(f"Error in batch: {e}")
    return processed, total



# ----------------------------
# PyQt5 GUI Implementation
# ----------------------------

# --- Mapping helper functions ---


def map_slider_to_multiplier(slider_value, min_multiplier=0.5, max_multiplier=1.5):
    """
    Map a slider value (0 to 100) to a multiplier between min_multiplier and max_multiplier.
    A value of 50 yields a neutral multiplier (1.0).
    """
    return min_multiplier + (max_multiplier - min_multiplier) * (slider_value / 100.0)


def map_slider_to_blur_radius(slider_value, max_radius=5):
    """
    Map a slider value (0 to 100) to a blur radius.
    A value of 50 could be considered moderate (half of max_radius).
    """
    return max_radius * (slider_value / 100.0)


# --- Enhanced Filter Functions ---


def apply_filter(pil_img, filter_name, slider_value=50):
    """
    Apply a filter to a PIL image using a slider_value for fine tuning.
    slider_value is expected to be in the range 0 to 100, with 50 as the neutral value.
    Supported filters: Brightness, Contrast, Saturation, Sharpness, Blur,
    Edge Detection, and Sepia.
    """
    # For brightness, contrast, saturation, and sharpness, map slider to a multiplier.
    # For example, slider_value=50 maps to 1.0 (neutral), while 0 maps to 0.5 and 100 to 1.5.
    brightness = lambda img: ImageEnhance.Brightness(img).enhance(
        map_slider_to_multiplier(slider_value, 0.5, 1.5)
    )
    contrast = lambda img: ImageEnhance.Contrast(img).enhance(
        map_slider_to_multiplier(slider_value, 0.5, 1.5)
    )
    saturation = lambda img: ImageEnhance.Color(img).enhance(
        map_slider_to_multiplier(slider_value, 0.5, 1.5)
    )
    sharpness = lambda img: ImageEnhance.Sharpness(img).enhance(
        map_slider_to_multiplier(slider_value, 0.5, 1.5)
    )
    # For blur, map the slider to a blur radius (e.g., 0 to 5)
    blur = lambda img: img.filter(
        ImageFilter.GaussianBlur(radius=map_slider_to_blur_radius(slider_value, 5))
    )
    # Edge detection remains binary; intensity is not applicable
    edge_detection = lambda img: img.filter(ImageFilter.FIND_EDGES)
    # Sepia: blend original with a sepia-toned version based on a normalized slider
    sepia = lambda img: apply_sepia(img, slider_value / 100.0)

    filter_functions = {
        "Brightness": brightness,
        "Contrast": contrast,
        "Saturation": saturation,
        "Sharpness": sharpness,
        "Blur": blur,
        "Edge Detection": edge_detection,
        "Sepia": sepia,
    }

    # Return the filtered image or the original if the filter is not found.
    return filter_functions.get(filter_name, lambda img: img)(pil_img)


def apply_sepia(pil_img, blend_factor=0.5):
    """
    Apply a sepia filter by blending the original image with a sepia-toned version.
    blend_factor should be between 0 (original) and 1 (full sepia).
    """
    # Convert image to grayscale
    grayscale = pil_img.convert("L")
    # Create a sepia-toned image via colorization
    sepia_img = ImageOps.colorize(grayscale, "#704214", "#C0A080")
    # Blend original and sepia images based on blend_factor
    return Image.blend(pil_img, sepia_img, blend_factor)


# --- Background Removal with Transparency ---


def remove_background_transparent(cv_img):
    """
    Remove the background from a CV2 image using GrabCut and output an image with transparency.
    The foreground pixels become fully opaque, and background pixels become transparent.
    """
    mask = np.zeros(cv_img.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    rect = (50, 50, cv_img.shape[1] - 50, cv_img.shape[0] - 50)
    cv2.grabCut(cv_img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    alpha = mask2 * 255
    b, g, r = cv2.split(cv_img)
    cv_img_transparent = cv2.merge([b, g, r, alpha])
    return cv_img_transparent


def apply_aspect_ratio_filter(pil_img, target_ratio):
    """
    Crop the PIL image to a target aspect ratio while keeping the crop centered.
    
    Args:
        pil_img (PIL.Image): The input image.
        target_ratio (float): Desired aspect ratio (width / height).
        
    Returns:
        PIL.Image: The cropped image.
    """
    width, height = pil_img.size
    current_ratio = width / height

    if current_ratio > target_ratio:
        # Image is too wide: crop the sides
        new_width = int(height * target_ratio)
        left = (width - new_width) // 2
        right = left + new_width
        crop_box = (left, 0, right, height)
    else:
        # Image is too tall: crop the top and bottom
        new_height = int(width / target_ratio)
        top = (height - new_height) // 2
        bottom = top + new_height
        crop_box = (0, top, width, bottom)
    
    return pil_img.crop(crop_box)



class Application(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Cropper")
        self.resize(600, 700)  # increased height to accommodate preview and controls
        self.init_ui()
        create_required_folders()
        # Variables to store preview data
        self.current_pil_image = None
        self.current_landmarks = None
        self.current_box = None
        self.current_metadata = None
        self.worker = None

        # Set up a QTimer to throttle preview updates
        self.preview_timer = QTimer(self)
        self.preview_timer.setInterval(300)  # delay in milliseconds
        self.preview_timer.setSingleShot(True)
        self.preview_timer.timeout.connect(self.update_preview_now)

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()

        # Folder selection grid
        grid = QGridLayout()
        grid.addWidget(QLabel("Input Folder:"), 0, 0)
        self.input_folder_edit = QLineEdit()
        grid.addWidget(self.input_folder_edit, 0, 1)
        btn_input = QPushButton("Browse")
        btn_input.clicked.connect(self.select_input_folder)
        grid.addWidget(btn_input, 0, 2)

        grid.addWidget(QLabel("Output Folder:"), 1, 0)
        self.output_folder_edit = QLineEdit()
        grid.addWidget(self.output_folder_edit, 1, 1)
        btn_output = QPushButton("Browse")
        btn_output.clicked.connect(self.select_output_folder)
        grid.addWidget(btn_output, 1, 2)
        layout.addLayout(grid)

        # Parameters (Margins)
        params_layout = QHBoxLayout()
        params_layout.addWidget(QLabel("Frontal Margin (px):"))
        self.margin_edit = QLineEdit("20")
        params_layout.addWidget(self.margin_edit)
        params_layout.addWidget(QLabel("Profile Margin (px):"))
        self.side_trim_edit = QLineEdit("20")
        params_layout.addWidget(self.side_trim_edit)
        layout.addLayout(params_layout)

        # Connect margin edits to trigger the preview timer (instead of immediate update)
        self.margin_edit.textChanged.connect(self.restart_preview_timer)
        self.side_trim_edit.textChanged.connect(self.restart_preview_timer)

        # Checkboxes for additional options
        options_layout = QHBoxLayout()
        self.sharpen_checkbox = QCheckBox("Sharpen Image")
        self.sharpen_checkbox.setChecked(True)
        options_layout.addWidget(self.sharpen_checkbox)
        self.frontal_checkbox = QCheckBox("Use Frontal Cropping")
        self.frontal_checkbox.setChecked(True)
        options_layout.addWidget(self.frontal_checkbox)
        self.profile_checkbox = QCheckBox("Use Profile Cropping")
        self.profile_checkbox.setChecked(True)
        options_layout.addWidget(self.profile_checkbox)
        # Connect checkbox changes to trigger preview updates
        self.frontal_checkbox.stateChanged.connect(self.restart_preview_timer)
        self.profile_checkbox.stateChanged.connect(self.restart_preview_timer)
        layout.addLayout(options_layout)
        self.rotation_checkbox = QCheckBox("Correct Face Rotation")
        self.rotation_checkbox.setChecked(True)
        options_layout.addWidget(self.rotation_checkbox)

        # New Crop Style Controls
        crop_style_layout = QHBoxLayout()
        crop_style_layout.addWidget(QLabel("Crop Style:"))
        self.crop_style_combo = QComboBox()
        self.crop_style_combo.addItems(
            ["auto", "frontal", "profile", "chin", "nose", "below_lips"]
        )
        self.crop_style_combo.currentTextChanged.connect(self.restart_preview_timer)
        crop_style_layout.addWidget(self.crop_style_combo)
        layout.addLayout(crop_style_layout)

        # Filter controls
        filter_layout = QFormLayout()
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(
            ["None", "Brightness", "Contrast", "Blur", "Edge Detection"]
        )
        self.filter_combo.currentTextChanged.connect(self.restart_preview_timer)
        filter_layout.addRow("Filter:", self.filter_combo)

        self.intensity_slider = QSlider(Qt.Horizontal)
        self.intensity_slider.setRange(0, 100)
        self.intensity_slider.setValue(50)
        self.intensity_slider.valueChanged.connect(self.restart_preview_timer)
        filter_layout.addRow("Intensity:", self.intensity_slider)
        
        # Aspect Ratio Dropdown for cropping
        self.aspect_ratio_combo = QComboBox()
        self.aspect_ratio_combo.addItems(["3:2", "4:3", "16:9"])
        self.aspect_ratio_combo.currentTextChanged.connect(self.restart_preview_timer)
        filter_layout.addRow("Aspect Ratio:", self.aspect_ratio_combo)
        
        layout.addLayout(filter_layout)

        # Progress display
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # Buttons
        btn_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Processing")
        self.start_button.clicked.connect(self.start_processing)
        btn_layout.addWidget(self.start_button)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.cancel_processing)
        btn_layout.addWidget(cancel_button)
        layout.addLayout(btn_layout)

        # Preview display area
        self.preview_label = QLabel("Preview will appear here")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setFixedSize(600, 400)
        layout.addWidget(self.preview_label, alignment=Qt.AlignCenter)

        # Button to load a preview image
        self.preview_button = QPushButton("Load Preview")
        self.preview_button.clicked.connect(self.load_preview)
        layout.addWidget(self.preview_button)

        central_widget.setLayout(layout)

    def restart_preview_timer(self):
        # Restart the timer every time a parameter changes.
        self.preview_timer.start()

    def select_input_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if folder:
            self.input_folder_edit.setText(folder)

    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_folder_edit.setText(folder)

    def update_progress(self, current, total, message):
        progress = int((current / total) * 100) if total > 0 else 0
        self.progress_bar.setValue(progress)
        self.status_label.setText(f"Processed {current}/{total} images. {message}")

    def pil_to_pixmap(self, pil_img):
        if pil_img.mode != "RGBA":
            pil_img = pil_img.convert("RGBA")
        data = pil_img.tobytes("raw", "RGBA")
        qimage = QImage(data, pil_img.width, pil_img.height, QImage.Format_RGBA8888)
        return QPixmap.fromImage(qimage)

    def load_preview(self):
        input_folder = self.input_folder_edit.text().strip()
        if not input_folder:
            QMessageBox.critical(self, "Error", "Please select an input folder first.")
            return
        valid_exts = (".jpg", ".jpeg", ".png", ".heic")
        files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid_exts)]
        if not files:
            QMessageBox.critical(
                self, "Error", "No valid image files found in the input folder."
            )
            return
        file_path = os.path.join(input_folder, files[0])
        result = get_face_and_landmarks(
            file_path,
            sharpen=self.sharpen_checkbox.isChecked(),
            apply_rotation=self.rotation_checkbox.isChecked(),
        )
        if result is None or result[0] is None or result[1] is None:
            QMessageBox.critical(self, "Error", "Failed to process image for preview.")
            return
        box, landmarks, cv_img, pil_img, metadata = result
        self.current_pil_image = pil_img
        self.current_landmarks = landmarks
        self.current_box = box
        self.current_metadata = metadata
        self.update_preview_now()

    def update_preview_now(self):
        if not self.current_pil_image:
            return
        try:
            frontal_margin = int(self.margin_edit.text())
            profile_margin = int(self.side_trim_edit.text())
        except ValueError:
            return

        crop_style = self.crop_style_combo.currentText()
        # Use a dictionary to map crop style to its lambda:
        crop_funcs = {
            "frontal": lambda: (
                crop_frontal_image_preview(
                    self.current_pil_image,
                    frontal_margin,
                    self.current_landmarks,
                    self.current_metadata,
                    lip_offset=50,
                )
                if self.frontal_checkbox.isChecked() and self.current_landmarks
                else None
            ),
            "profile": lambda: (
                crop_profile_image_preview(
                    self.current_pil_image,
                    profile_margin,
                    50,
                    self.current_box,
                    self.current_metadata,
                )
                if self.profile_checkbox.isChecked() and self.current_box
                else None
            ),
            "chin": lambda: crop_chin_image(
                self.current_pil_image,
                frontal_margin,
                self.current_box,
                self.current_metadata,
                chin_offset=20,
            ),
            "nose": lambda: crop_nose_image(
                self.current_pil_image,
                self.current_box,
                self.current_landmarks,
                self.current_metadata,
                margin=0,
            ),
            "below_lips": lambda: crop_below_lips_image(
                self.current_pil_image,
                frontal_margin,
                self.current_landmarks,
                self.current_metadata,
                offset=10,
            ),
            "auto": lambda: auto_crop(
                self.current_pil_image,
                frontal_margin,
                profile_margin,
                self.current_box,
                self.current_landmarks,
                self.current_metadata,
            ),
        }
        cropped_img = crop_funcs.get(crop_style, lambda: None)()
        
        # Retrieve selected aspect ratio and enforce it if a crop exists
        selected_ratio = self.aspect_ratio_combo.currentText()
        if selected_ratio == "3:2":
            target_ratio = 3 / 2
        elif selected_ratio == "4:3":
            target_ratio = 4 / 3
        elif selected_ratio == "16:9":
            target_ratio = 16 / 9
        else:
            target_ratio = None

        if cropped_img and target_ratio:
            cropped_img = apply_aspect_ratio_filter(cropped_img, target_ratio)

        if cropped_img:
            filter_name = self.filter_combo.currentText()
            intensity = self.intensity_slider.value()
            cropped_img = apply_filter(cropped_img, filter_name, intensity)
            pixmap = self.pil_to_pixmap(cropped_img)
            scaled_pixmap = pixmap.scaled(
                self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.preview_label.setPixmap(scaled_pixmap)
        else:
            self.preview_label.setText("No preview available.")


    def start_processing(self):
        input_folder = self.input_folder_edit.text().strip()
        output_folder = self.output_folder_edit.text().strip()
        try:
            frontal_margin = int(self.margin_edit.text())
            profile_margin = int(self.side_trim_edit.text())
        except ValueError:
            QMessageBox.critical(
                self, "Error", "Margin and Profile Margin must be integers."
            )
            return
        if not input_folder or not output_folder:
            QMessageBox.critical(
                self, "Error", "Please select both input and output folders."
            )
            return

        # Get selected aspect ratio from the dropdown
        selected_ratio = self.aspect_ratio_combo.currentText()
        if selected_ratio == "3:2":
            aspect_ratio = 3 / 2
        elif selected_ratio == "4:3":
            aspect_ratio = 4 / 3
        elif selected_ratio == "16:9":
            aspect_ratio = 16 / 9
        else:
            aspect_ratio = None

        sharpen = self.sharpen_checkbox.isChecked()
        use_frontal = self.frontal_checkbox.isChecked()
        use_profile = self.profile_checkbox.isChecked()
        crop_style = self.crop_style_combo.currentText()

        self.start_button.setEnabled(False)
        self.thread = QThread()
        self.worker = Worker(
            input_folder,
            output_folder,
            frontal_margin,
            profile_margin,
            sharpen,
            use_frontal,
            use_profile,
            self.rotation_checkbox.isChecked(),
            crop_style,
            self.filter_combo.currentText(),
            self.intensity_slider.value(),
            aspect_ratio  # Pass the aspect ratio here
        )
        self.worker.moveToThread(self.thread)
        self.worker.progress_update.connect(self.update_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.thread.started.connect(self.worker.run)
        self.thread.start()

    def cancel_processing(self):
        if self.worker:
            self.worker.cancel()
            self.status_label.setText("Cancelling processing...")
            self.start_button.setEnabled(False)

    def on_finished(self, processed, total):
        self.status_label.setText(f"Successfully processed {processed}/{total} images")
        self.progress_bar.setValue(100)
        QMessageBox.information(
            self, "Complete", f"Processed {processed} of {total} images"
        )
        self.start_button.setEnabled(True)
        self.thread.quit()
        self.thread.wait()
        self.worker = None

    def on_error(self, error_message):
        QMessageBox.critical(self, "Error", error_message)
        self.start_button.setEnabled(True)
        self.thread.quit()
        self.thread.wait()
        self.worker = None


class Worker(QObject):
    finished = pyqtSignal(int, int)
    progress_update = pyqtSignal(int, int, str)
    error = pyqtSignal(str)

    def __init__(
        self,
        input_folder,
        output_folder,
        frontal_margin,
        profile_margin,
        sharpen,
        use_frontal,
        use_profile,
        correct_rotation,
        crop_style,
        filter_name,
        filter_intensity,
        aspect_ratio  # New parameter
    ):
        super().__init__()
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.frontal_margin = frontal_margin
        self.profile_margin = profile_margin
        self.sharpen = sharpen
        self.use_frontal = use_frontal
        self.use_profile = use_profile
        self.correct_rotation = correct_rotation
        self.crop_style = crop_style
        self.filter_name = filter_name
        self.filter_intensity = filter_intensity
        self.aspect_ratio = aspect_ratio
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def is_cancelled(self):
        return self._cancelled

    def run(self):
        try:
            processed, total = process_images_threaded(
                self.input_folder,
                self.output_folder,
                self.frontal_margin,
                self.profile_margin,
                self.sharpen,
                self.use_frontal,
                self.use_profile,
                self.progress_update.emit,
                cancel_func=self.is_cancelled,
                apply_rotation=self.correct_rotation,
                crop_style=self.crop_style,
                filter_name=self.filter_name,
                filter_intensity=self.filter_intensity,
                aspect_ratio=self.aspect_ratio
            )
            if self.is_cancelled():
                self.error.emit("Processing was cancelled.")
            else:
                self.finished.emit(processed, total)
        except Exception as e:
            self.error.emit(str(e))


# --- Main entry point remains unchanged ---
if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    app = QApplication(sys.argv)
    window = Application()
    window.show()
    exit_code = app.exec_()

    profiler.disable()
    with open("profile_output.txt", "w") as f:
        stats = pstats.Stats(profiler, stream=f)
        stats.sort_stats("cumtime")
        stats.print_stats()
    print("Profile saved to profile_output.txt")
    sys.exit(exit_code)
