from __future__ import annotations

from pathlib import Path
import uuid
import importlib.metadata

# Monkey patch to handle missing packages (like torchvision) gracefully
_original_version = importlib.metadata.version

def _patched_version(name):
    try:
        return _original_version(name)
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0"

importlib.metadata.version = _patched_version

import cv2
import numpy as np
from flask import Flask, redirect, render_template, request, url_for
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
UPLOAD_FOLDER = ROOT / "uploads"
RESULT_FOLDER = ROOT / "static" / "results"
RUNS_FOLDER = ROOT / "runs"
FALLBACK_MODEL_PATH = ROOT / "yolov8n-cls.pt"

MIN_CONFIDENCE = 0.55
MIN_MARGIN = 0.10
FACE_PADDING = 0.25

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)

EXPRESSION_EMOJI = {
    "happy": ":)",
    "sad": ":(",
    "angry": ">:(",
    "surprise": ":O",
    "fear": ":/",
    "disgust": ":-P",
    "neutral": ":|",
    "uncertain": "?",
}

EXPRESSION_COLOR = {
    "happy": "#f9c74f",
    "sad": "#4cc9f0",
    "angry": "#f72585",
    "surprise": "#ff9f1c",
    "fear": "#b5179e",
    "disgust": "#76c893",
    "neutral": "#adb5bd",
    "uncertain": "#ffd166",
}


def resolve_model_path() -> Path:
    candidates: list[Path] = []

    root_best = ROOT / "best.pt"
    if root_best.exists():
        candidates.append(root_best)

    if RUNS_FOLDER.exists():
        candidates.extend(path for path in RUNS_FOLDER.rglob("weights/best.pt") if path.is_file())

    if candidates:
        return max(candidates, key=lambda path: path.stat().st_mtime)

    return FALLBACK_MODEL_PATH


MODEL_PATH = resolve_model_path()
model = YOLO(str(MODEL_PATH))
MODEL_SOURCE = "custom" if MODEL_PATH != FALLBACK_MODEL_PATH else "base"
print(f"[OK] Loaded {MODEL_SOURCE} model: {MODEL_PATH}")

FACE_CASCADE = cv2.CascadeClassifier(
    str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
)


@app.context_processor
def inject_model_meta() -> dict[str, str]:
    badge = "YOLOv8 · Custom trained" if MODEL_SOURCE == "custom" else "YOLOv8 · Base model"
    return {"model_badge": badge, "model_name": MODEL_PATH.name}


def class_name(index: int) -> str:
    return model.names[int(index)]


def detect_primary_face(image_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    if FACE_CASCADE.empty():
        return None

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    detection_configs = (
        (gray, 1.10, 6),
        (clahe, 1.08, 5),
        (clahe, 1.05, 4),
    )

    faces: list[tuple[int, int, int, int]] = []
    for source, scale_factor, min_neighbors in detection_configs:
        detected = FACE_CASCADE.detectMultiScale(
            source,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=(32, 32),
        )
        if len(detected):
            faces = [tuple(map(int, face)) for face in detected]
            break

    if not faces:
        return None

    return max(faces, key=lambda face: face[2] * face[3])


def expand_face_box(
    face_box: tuple[int, int, int, int], width: int, height: int, padding: float = FACE_PADDING
) -> tuple[int, int, int, int]:
    x, y, w, h = face_box
    pad_x = int(w * padding)
    pad_y = int(h * padding)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(width, x + w + pad_x)
    y2 = min(height, y + h + pad_y)
    return x1, y1, x2, y2


def crop_face(image_bgr: np.ndarray, face_box: tuple[int, int, int, int] | None) -> np.ndarray:
    if face_box is None:
        return image_bgr

    height, width = image_bgr.shape[:2]
    x1, y1, x2, y2 = expand_face_box(face_box, width, height)
    return image_bgr[y1:y2, x1:x2]


def create_display_image(
    image_bgr: np.ndarray, face_box: tuple[int, int, int, int] | None, output_path: Path
) -> None:
    display = image_bgr.copy()

    if face_box is not None:
        x, y, w, h = face_box
        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 229, 255), 2)
        cv2.putText(
            display,
            "face used",
            (x, max(24, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 229, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), display)


def predict_probabilities(image_bgr: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    flipped = np.ascontiguousarray(np.fliplr(rgb))
    sources = [rgb, flipped]
    results = model.predict(source=sources, verbose=False)

    probabilities = [
        np.asarray(result.probs.data.detach().cpu(), dtype=np.float32) for result in results
    ]
    return np.mean(np.stack(probabilities, axis=0), axis=0)


def build_top_predictions(probabilities: np.ndarray) -> list[dict[str, object]]:
    top_indices = np.argsort(probabilities)[::-1][:5]
    predictions: list[dict[str, object]] = []
    for index in top_indices:
        label = class_name(int(index))
        predictions.append(
            {
                "label": label,
                "confidence": round(float(probabilities[index]) * 100, 2),
                "emoji": EXPRESSION_EMOJI.get(label, "?"),
                "color": EXPRESSION_COLOR.get(label, "#ffffff"),
            }
        )
    return predictions


def summarize_prediction(
    probabilities: np.ndarray, face_detected: bool
) -> tuple[str, float, str, str, str]:
    top_indices = np.argsort(probabilities)[::-1]
    top_index = int(top_indices[0])
    second_index = int(top_indices[1]) if len(top_indices) > 1 else top_index

    top_label = class_name(top_index)
    second_label = class_name(second_index)
    top_confidence = float(probabilities[top_index])
    second_confidence = float(probabilities[second_index])
    confidence_gap = top_confidence - second_confidence
    required_confidence = MIN_CONFIDENCE if face_detected else 0.65
    required_margin = MIN_MARGIN if face_detected else 0.15

    if top_confidence >= required_confidence and confidence_gap >= required_margin:
        note = "Largest detected face was used before classification." if face_detected else (
            "No face crop was available, so the full image was analyzed."
        )
        return (
            top_label,
            round(top_confidence * 100, 2),
            EXPRESSION_EMOJI.get(top_label, "?"),
            EXPRESSION_COLOR.get(top_label, "#ffffff"),
            note,
        )

    note = (
        f"Low-confidence result. Closest guesses were {top_label} ({top_confidence * 100:.1f}%) "
        f"and {second_label} ({second_confidence * 100:.1f}%). "
        "Try a brighter, front-facing photo with one clear face."
    )
    if not face_detected:
        note = "No clear face was detected. " + note

    return (
        "uncertain",
        round(top_confidence * 100, 2),
        EXPRESSION_EMOJI["uncertain"],
        EXPRESSION_COLOR["uncertain"],
        note,
    )


def analyze_image(image_path: Path) -> dict[str, object]:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    face_box = detect_primary_face(image_bgr)
    face_crop = crop_face(image_bgr, face_box)
    probabilities = predict_probabilities(face_crop)
    top5 = build_top_predictions(probabilities)
    top_label, top_conf, emoji, color, note = summarize_prediction(
        probabilities, face_detected=face_box is not None
    )

    display_name = f"{uuid.uuid4().hex}.jpg"
    display_path = RESULT_FOLDER / display_name
    create_display_image(image_bgr, face_box, display_path)

    return {
        "image": f"/static/results/{display_name}",
        "top_label": top_label,
        "top_conf": top_conf,
        "top5": top5,
        "emoji": emoji,
        "color": color,
        "result_note": note,
    }


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return redirect(url_for("home"))

    file = request.files["image"]
    if file.filename == "":
        return redirect(url_for("home"))

    upload_name = f"{uuid.uuid4().hex}{Path(file.filename).suffix.lower() or '.jpg'}"
    upload_path = UPLOAD_FOLDER / upload_name
    file.save(str(upload_path))

    try:
        result = analyze_image(upload_path)
    except Exception as error:
        if upload_path.exists():
            upload_path.unlink()
        return render_template(
            "index.html",
            image=None,
            result_note=f"Could not analyze that image: {error}",
        )

    if upload_path.exists():
        upload_path.unlink()

    return render_template("index.html", **result)


if __name__ == "__main__":
    app.run(debug=True)
