"""Flask app for low-light Singapore number plate detection and review."""

from __future__ import annotations

import base64
import re
import ssl
import threading
import webbrowser
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from fast_alpr import ALPR
from fast_plate_ocr import LicensePlateRecognizer
from flask import Flask, jsonify, render_template, request
from ultralytics import YOLO

from step3_heuristics import (
    checksum_corrected_plate,
    clean_plate_text,
    format_plate_for_display,
    normalize_singapore_plate,
    status_from_score,
)


ssl._create_default_https_context = ssl._create_unverified_context

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
BEST_MODEL_PATH = BASE_DIR / "best.pt"
UPLOAD_FOLDER = BASE_DIR / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "bmp"}
DETECTION_IMGSZ = 960
DETECTION_CONF = 0.15
# Focused top-half crops often contain the rear plate but score lower in tunnel images,
# so we allow a softer detector threshold on those passes.
FOCUSED_DETECTION_CONF = 0.05
MAX_RESULTS = 3
GENERIC_PLATE_RE = re.compile(r"^(?=.*[A-Z])(?=.*\d)[A-Z0-9]{4,10}$")


@dataclass(frozen=True)
class ImageSource:
    name: str
    image: np.ndarray
    offset_x: int = 0
    offset_y: int = 0
    scale_x: float = 1.0
    scale_y: float = 1.0
    focus_bonus: float = 0.0


plate_detector: YOLO | None = None
plate_ocr: LicensePlateRecognizer | None = None
fallback_alpr: ALPR | None = None
model_errors: list[str] = []


def load_models() -> None:
    global plate_detector, plate_ocr, fallback_alpr, model_errors

    model_errors = []

    print("Initializing custom detector...")
    try:
        if not BEST_MODEL_PATH.exists():
            raise FileNotFoundError(f"Missing trained model: {BEST_MODEL_PATH}")
        plate_detector = YOLO(str(BEST_MODEL_PATH))
        print("Custom detector ready")
    except Exception as exc:
        plate_detector = None
        model_errors.append(f"Custom detector failed: {exc}")
        print(model_errors[-1])

    print("Initializing OCR...")
    try:
        plate_ocr = LicensePlateRecognizer(
            hub_ocr_model="cct-s-v2-global-model",
            device="auto",
        )
        print("OCR ready")
    except Exception as exc:
        plate_ocr = None
        model_errors.append(f"OCR failed: {exc}")
        print(model_errors[-1])

    print("Initializing fallback ALPR...")
    try:
        fallback_alpr = ALPR(
            detector_model="yolo-v9-t-384-license-plate-end2end",
            ocr_model="cct-xs-v2-global-model",
        )
        print("Fallback ALPR ready")
    except Exception as exc:
        fallback_alpr = None
        model_errors.append(f"Fallback ALPR failed: {exc}")
        print(model_errors[-1])


load_models()


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def encode_image(img: np.ndarray) -> str:
    _, buffer = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buffer).decode()


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def mean_character_confidence(text: str, confidences: object) -> float:
    cleaned = clean_plate_text(text)
    if not cleaned or confidences is None:
        return 0.0

    if isinstance(confidences, (float, int)):
        return clamp(float(confidences))

    scores = np.asarray(confidences, dtype=float).flatten()
    if scores.size == 0:
        return 0.0

    length = min(len(cleaned), scores.size)
    if length == 0:
        return 0.0

    return clamp(float(scores[:length].mean()))


def normalize_generic_plate_text(text: str) -> str:
    cleaned = clean_plate_text(text)
    if not GENERIC_PLATE_RE.match(cleaned):
        return ""
    return cleaned


def sharpen(img: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(img, (0, 0), 2)
    return cv2.addWeighted(img, 1.6, blur, -0.6, 0)


def gamma_correct(img: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    inverse_gamma = 1.0 / gamma
    table = np.array(
        [((value / 255.0) ** inverse_gamma) * 255 for value in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(img, table)


def clahe_bgr(img: np.ndarray, clip_limit: float = 4.0, tiles: tuple[int, int] = (6, 6)) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_channel = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tiles).apply(l_channel)
    return cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)


def enhance_for_detection(img: np.ndarray) -> np.ndarray:
    boosted = gamma_correct(img, gamma=2.4)
    boosted = clahe_bgr(boosted, clip_limit=4.5, tiles=(6, 6))
    return sharpen(boosted)


def enhance_lowlight(img: np.ndarray) -> np.ndarray:
    upscaled = cv2.resize(img, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_LANCZOS4)
    boosted = clahe_bgr(upscaled, clip_limit=3.8, tiles=(8, 8))
    return sharpen(boosted)


def enhance_dark(img: np.ndarray) -> np.ndarray:
    boosted = gamma_correct(img, gamma=2.5)
    boosted = cv2.resize(boosted, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_LANCZOS4)
    boosted = clahe_bgr(boosted, clip_limit=5.0, tiles=(6, 6))
    return sharpen(boosted)


def crop_upper(img: np.ndarray) -> np.ndarray:
    return cv2.resize(img[: img.shape[0] // 2, :], None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LANCZOS4)


def crop_upper_dark(img: np.ndarray) -> np.ndarray:
    return enhance_dark(img[: img.shape[0] // 2, :])


def build_primary_sources(img: np.ndarray) -> list[ImageSource]:
    top_half = img[: img.shape[0] // 2, :]
    return [
        ImageSource(name="top_half", image=top_half, focus_bonus=0.08),
        ImageSource(name="top_half_enhanced", image=enhance_for_detection(top_half), focus_bonus=0.10),
        ImageSource(name="full_frame", image=img, focus_bonus=0.0),
        ImageSource(name="full_frame_enhanced", image=enhance_for_detection(img), focus_bonus=0.02),
    ]


def build_fallback_sources(img: np.ndarray) -> list[ImageSource]:
    return [
        ImageSource(name="fallback_full", image=img, focus_bonus=0.0),
        ImageSource(name="fallback_sharp", image=sharpen(img), focus_bonus=0.02),
        ImageSource(name="fallback_lowlight", image=enhance_lowlight(img), scale_x=2.5, scale_y=2.5, focus_bonus=0.03),
        ImageSource(name="fallback_upper", image=crop_upper(img), scale_x=2.0, scale_y=2.0, focus_bonus=0.08),
        ImageSource(name="fallback_upper_dark", image=crop_upper_dark(img), scale_x=2.5, scale_y=2.5, focus_bonus=0.10),
    ]


def build_ocr_variants(crop: np.ndarray) -> dict[str, np.ndarray]:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return {
        "orig": crop,
        "up2": cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC),
        "sharp": sharpen(crop),
        "clahe": clahe_bgr(crop, clip_limit=4.0, tiles=(6, 6)),
        "gray_clahe": cv2.cvtColor(gray_clahe, cv2.COLOR_GRAY2BGR),
        "gray_clahe_up3": cv2.cvtColor(
            cv2.resize(gray_clahe, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC),
            cv2.COLOR_GRAY2BGR,
        ),
        "gamma": gamma_correct(crop, gamma=2.2),
        "gamma_clahe": clahe_bgr(gamma_correct(crop, gamma=2.4), clip_limit=4.5, tiles=(6, 6)),
        "gamma_clahe_up2": cv2.resize(
            clahe_bgr(gamma_correct(crop, gamma=2.4), clip_limit=4.5, tiles=(6, 6)),
            None,
            fx=2.0,
            fy=2.0,
            interpolation=cv2.INTER_CUBIC,
        ),
    }


def map_box_to_original(box_xyxy: tuple[float, float, float, float], source: ImageSource) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box_xyxy
    return (
        int(round(x1 / source.scale_x)) + source.offset_x,
        int(round(y1 / source.scale_y)) + source.offset_y,
        int(round(x2 / source.scale_x)) + source.offset_x,
        int(round(y2 / source.scale_y)) + source.offset_y,
    )


def clip_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))
    return x1, y1, x2, y2


def valid_box(box: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    return width >= 50 and height >= 15 and (width * height) >= 900


def crop_with_padding(img: np.ndarray, box: tuple[int, int, int, int], pad_x_ratio: float = 0.12, pad_y_ratio: float = 0.25) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    x1, y1, x2, y2 = box
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)

    pad_x = int(width * pad_x_ratio)
    pad_y = int(height * pad_y_ratio)

    cropped_box = clip_box(
        (x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y),
        img.shape[1],
        img.shape[0],
    )
    cx1, cy1, cx2, cy2 = cropped_box
    return img[cy1:cy2, cx1:cx2], cropped_box


def position_bonus(box: tuple[int, int, int, int], img_height: int) -> float:
    center_y = (box[1] + box[3]) / 2.0
    ratio = center_y / max(img_height, 1)
    if ratio <= 0.55:
        return 0.05
    if ratio >= 0.68:
        return -0.14
    return 0.0


def primary_pipeline_available() -> bool:
    return plate_detector is not None and plate_ocr is not None


def run_primary_pipeline(img: np.ndarray) -> list[dict]:
    if not primary_pipeline_available():
        return []

    candidates: list[dict] = []
    img_height = img.shape[0]

    for source in build_primary_sources(img):
        detector_conf_threshold = DETECTION_CONF
        if source.name.startswith("top_half"):
            detector_conf_threshold = min(DETECTION_CONF, FOCUSED_DETECTION_CONF)

        prediction = plate_detector.predict(
            source.image,
            imgsz=DETECTION_IMGSZ,
            conf=detector_conf_threshold,
            iou=0.45,
            max_det=8,
            verbose=False,
        )[0]

        if prediction.boxes is None:
            continue

        for detected_box in prediction.boxes:
            det_conf = clamp(safe_float(detected_box.conf[0].item()))
            mapped_box = map_box_to_original(tuple(detected_box.xyxy[0].tolist()), source)
            mapped_box = clip_box(mapped_box, img.shape[1], img.shape[0])

            if not valid_box(mapped_box):
                continue

            crop, padded_box = crop_with_padding(img, mapped_box)
            candidate = run_crop_ocr(
                crop=crop,
                padded_box=padded_box,
                detection_confidence=det_conf,
                source_name=source.name,
                focus_bonus=source.focus_bonus + position_bonus(mapped_box, img_height),
                method="best_pt",
            )
            if candidate:
                candidates.append(candidate)

    return candidates


def run_crop_ocr(
    crop: np.ndarray,
    padded_box: tuple[int, int, int, int],
    detection_confidence: float,
    source_name: str,
    focus_bonus: float,
    method: str,
) -> dict | None:
    grouped: dict[str, dict] = {}

    for variant_name, variant in build_ocr_variants(crop).items():
        try:
            rgb_variant = cv2.cvtColor(variant, cv2.COLOR_BGR2RGB)
            prediction = plate_ocr.run_one(rgb_variant, return_confidence=True)
        except Exception:
            continue

        normalized = normalize_singapore_plate(prediction.plate)
        generic_plate = normalize_generic_plate_text(prediction.plate)
        if not normalized.valid_shape and not generic_plate:
            continue

        ocr_confidence = mean_character_confidence(prediction.plate, prediction.char_probs)
        if ocr_confidence <= 0.0:
            continue

        if normalized.valid_shape:
            plate_key = normalized.compact
            display_plate = format_plate_for_display(normalized.pretty)
            checksum_match = normalized.checksum_match
            checksum_expected = normalized.checksum_expected
            format_score = normalized.format_score
            plate_body = f"{normalized.prefix}{normalized.digits}"
            is_singapore_plate = True
            non_sg_note = ""
        else:
            plate_key = generic_plate
            display_plate = generic_plate
            checksum_match = False
            checksum_expected = ""
            format_score = 0.38
            plate_body = generic_plate
            is_singapore_plate = False
            non_sg_note = "Detected plate text does not match Singapore format."

        if plate_key not in grouped:
            grouped[plate_key] = {
                "plate": plate_key,
                "display_plate": display_plate,
                "checksum_match": checksum_match,
                "checksum_expected": checksum_expected,
                "format_score": format_score,
                "plate_body": plate_body,
                "variant_hits": 0,
                "best_ocr_confidence": 0.0,
                "ocr_confidence_sum": 0.0,
                "source": source_name,
                "method": method,
                "box": padded_box,
                "is_singapore_plate": is_singapore_plate,
                "display_note": non_sg_note,
            }

        bucket = grouped[plate_key]
        bucket["variant_hits"] += 1
        bucket["best_ocr_confidence"] = max(bucket["best_ocr_confidence"], ocr_confidence)
        bucket["ocr_confidence_sum"] += ocr_confidence
        bucket.setdefault("variants", []).append(variant_name)

    if not grouped:
        return None

    ranked_candidates = []
    for candidate in grouped.values():
        average_ocr = candidate["ocr_confidence_sum"] / candidate["variant_hits"]
        consensus_score = min(candidate["variant_hits"] / 3.0, 1.0)
        if candidate.get("is_singapore_plate", True):
            final_score = (
                0.43 * candidate["best_ocr_confidence"]
                + 0.17 * average_ocr
                + 0.18 * detection_confidence
                + 0.17 * candidate["format_score"]
                + 0.05 * consensus_score
                + focus_bonus
            )
            final_score = clamp(final_score)

            if not candidate["checksum_match"]:
                final_score = min(final_score, 0.74)

            status, status_label = status_from_score(final_score, candidate["checksum_match"], candidate["variant_hits"])
        else:
            final_score = clamp(
                0.47 * candidate["best_ocr_confidence"]
                + 0.22 * average_ocr
                + 0.21 * detection_confidence
                + 0.05 * consensus_score
                + 0.05 * candidate["format_score"]
                + focus_bonus
            )
            status, status_label = "review", "Needs Review"

        candidate.update(
            {
                "confidence": final_score,
                "ocr_confidence": average_ocr,
                "detector_confidence": detection_confidence,
                "status": status,
                "status_label": status_label,
            }
        )
        ranked_candidates.append(candidate)

    ranked_candidates.sort(
        key=lambda item: (
            item["status"] == "confirmed",
            item["confidence"],
            item["checksum_match"],
            item["variant_hits"],
        ),
        reverse=True,
    )
    return ranked_candidates[0]


def run_fallback_pipeline(img: np.ndarray) -> list[dict]:
    if fallback_alpr is None:
        return []

    fallback_candidates: list[dict] = []
    img_height = img.shape[0]

    for source in build_fallback_sources(img):
        try:
            drawn = fallback_alpr.draw_predictions(source.image)
        except Exception:
            continue

        for result in drawn.results or []:
            ocr_result = getattr(result, "ocr", None)
            detection_result = getattr(result, "detection", None)
            raw_text = getattr(ocr_result, "text", "") if ocr_result else ""
            normalized = normalize_singapore_plate(raw_text)
            generic_plate = normalize_generic_plate_text(raw_text)
            if not normalized.valid_shape and not generic_plate:
                continue

            detector_confidence = clamp(getattr(detection_result, "confidence", 0.0) if detection_result else 0.0)
            ocr_confidence = mean_character_confidence(
                raw_text,
                getattr(ocr_result, "confidence", None) if ocr_result else None,
            )

            bounding_box = getattr(detection_result, "bounding_box", None)
            if bounding_box is None:
                continue

            mapped_box = map_box_to_original(
                (bounding_box.x1, bounding_box.y1, bounding_box.x2, bounding_box.y2),
                source,
            )
            mapped_box = clip_box(mapped_box, img.shape[1], img.shape[0])
            if not valid_box(mapped_box):
                continue

            if normalized.valid_shape:
                final_score = (
                    0.37 * ocr_confidence
                    + 0.17 * detector_confidence
                    + 0.31 * normalized.format_score
                    + source.focus_bonus
                    + position_bonus(mapped_box, img_height)
                )
                final_score = clamp(final_score)

                if not normalized.checksum_match:
                    final_score = min(final_score, 0.72)

                plate_text = normalized.compact
                display_plate = format_plate_for_display(normalized.pretty)
                checksum_match = normalized.checksum_match
                format_score = normalized.format_score
                is_singapore_plate = True
                display_note = ""
                status, status_label = status_from_score(final_score, normalized.checksum_match, 1)
            else:
                final_score = clamp(
                    0.46 * ocr_confidence
                    + 0.22 * detector_confidence
                    + 0.08 * source.focus_bonus
                    + 0.08 * max(position_bonus(mapped_box, img_height), -0.02)
                    + 0.16
                )
                plate_text = generic_plate
                display_plate = generic_plate
                checksum_match = False
                format_score = 0.38
                is_singapore_plate = False
                display_note = "Detected plate text does not match Singapore format."
                status, status_label = "review", "Needs Review"

            fallback_candidates.append(
                {
                    "plate": plate_text,
                    "display_plate": display_plate,
                    "confidence": final_score,
                    "ocr_confidence": ocr_confidence,
                    "detector_confidence": detector_confidence,
                    "checksum_match": checksum_match,
                    "format_score": format_score,
                    "variant_hits": 1,
                    "source": source.name,
                    "method": "fallback_alpr",
                    "box": mapped_box,
                    "status": status,
                    "status_label": status_label,
                    "is_singapore_plate": is_singapore_plate,
                    "display_note": display_note,
                }
            )

    return fallback_candidates


def consolidate_candidates(candidates: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate["plate"], []).append(candidate)

    merged_candidates: list[dict] = []
    for plate_text, plate_items in grouped.items():
        plate_items.sort(key=lambda item: item["confidence"], reverse=True)
        best = dict(plate_items[0])
        support_count = len(plate_items)
        support_bonus = min((support_count - 1) * 0.04, 0.08)
        best["support_count"] = support_count
        best["confidence"] = clamp(best["confidence"] + support_bonus)
        best["variant_hits"] = max(item["variant_hits"] for item in plate_items)
        best["status"], best["status_label"] = status_from_score(
            best["confidence"],
            best["checksum_match"],
            max(best["variant_hits"], support_count),
        )
        refine_candidate_status(best)
        merged_candidates.append(best)

    merged_candidates.sort(
        key=lambda item: (
            item["status"] == "confirmed",
            item["confidence"],
            item["checksum_match"],
            item["support_count"],
        ),
        reverse=True,
    )
    return merged_candidates


def expand_checksum_corrections(candidates: list[dict]) -> list[dict]:
    expanded: list[dict] = []
    for candidate in candidates:
        expanded.append(candidate)
        corrected = build_checksum_correction_candidate(candidate)
        if corrected:
            expanded.append(corrected)

    expanded.sort(
        key=lambda item: (
            item["status"] == "confirmed",
            item["confidence"],
            item["checksum_match"],
            item["support_count"],
            not item.get("is_checksum_corrected", False),
        ),
        reverse=True,
    )
    return expanded


def refine_candidate_status(candidate: dict) -> None:
    isolated_full_frame_guess = (
        candidate["status"] == "confirmed"
        and candidate["support_count"] <= 2
        and candidate["method"] == "best_pt"
        and not candidate["source"].startswith("top_half")
        and candidate["detector_confidence"] < 0.65
    )

    if isolated_full_frame_guess:
        candidate["status"] = "review"
        candidate["status_label"] = "Needs Review"


def build_checksum_correction_candidate(candidate: dict) -> dict | None:
    if candidate["checksum_match"]:
        return None

    if candidate.get("support_count", 1) < 3:
        return None

    if candidate["detector_confidence"] < 0.85 or candidate.get("best_ocr_confidence", 0.0) < 0.85:
        return None

    if candidate.get("format_score", 0.0) < 0.68:
        return None

    normalized = normalize_singapore_plate(candidate["plate"])
    corrected_plate = checksum_corrected_plate(normalized)
    if not corrected_plate or corrected_plate == candidate["plate"]:
        return None

    corrected_candidate = dict(candidate)
    corrected_candidate["plate"] = corrected_plate
    corrected_candidate["display_plate"] = format_plate_for_display(corrected_plate)
    corrected_candidate["checksum_match"] = True
    corrected_candidate["confidence"] = clamp(min(candidate["confidence"], 0.79) + 0.03)
    corrected_candidate["status"] = "review"
    corrected_candidate["status_label"] = "Needs Review"
    corrected_candidate["method"] = f"{candidate['method']}_checksum"
    corrected_candidate["correction_note"] = f"Checksum-guided suggestion from {candidate['plate']}"
    corrected_candidate["is_checksum_corrected"] = True
    return corrected_candidate


def candidate_is_publishable(candidate: dict) -> bool:
    if not candidate.get("is_singapore_plate", True):
        return False

    if candidate["status"] == "confirmed":
        return True

    if candidate["status"] != "review":
        return False

    if candidate.get("is_checksum_corrected", False):
        return (
            candidate.get("support_count", 1) >= 3
            and candidate["detector_confidence"] >= 0.85
            and candidate.get("best_ocr_confidence", 0.0) >= 0.85
            and candidate["confidence"] >= 0.76
        )

    if not candidate["checksum_match"]:
        return False

    if candidate["method"] == "fallback_alpr":
        return candidate["confidence"] >= 0.72

    if candidate["source"].startswith("top_half"):
        return candidate["confidence"] >= 0.72

    return candidate["detector_confidence"] >= 0.70 and candidate["confidence"] >= 0.80


def candidate_is_visible_review(candidate: dict) -> bool:
    if candidate["status"] != "review":
        return False

    if candidate.get("is_singapore_plate", True):
        if candidate["confidence"] < 0.72:
            return False
    else:
        if candidate["confidence"] < 0.58:
            return False

    return True


def prepare_review_candidate_for_display(candidate: dict) -> dict:
    display_candidate = dict(candidate)
    display_candidate["suppressed_review"] = True
    display_candidate["status"] = "review"
    display_candidate["status_label"] = "Waiting for human confirmation"
    if display_candidate.get("is_singapore_plate", True):
        display_candidate["display_note"] = (
            "No plate passed strict validation, so the strongest review result is shown for manual confirmation."
        )
    else:
        display_candidate["display_note"] = (
            "Plate text was detected, but it does not match Singapore format, so it is shown for manual confirmation."
        )
    return display_candidate


def annotate_image(img: np.ndarray, candidates: list[dict]) -> np.ndarray:
    annotated = img.copy()
    for candidate in candidates:
        x1, y1, x2, y2 = candidate["box"]
        if candidate["status"] == "confirmed":
            color = (46, 204, 113)
        else:
            color = (0, 191, 255)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{candidate['plate']} | {int(candidate['confidence'] * 100)}%"
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        text_y = max(0, y1 - text_height - baseline - 8)
        cv2.rectangle(
            annotated,
            (x1, text_y),
            (x1 + text_width + 10, text_y + text_height + baseline + 10),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 5, text_y + text_height + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    return annotated


def process_image(img: np.ndarray) -> tuple[list[dict], np.ndarray]:
    all_candidates = run_primary_pipeline(img)
    has_confirmed_primary = any(candidate["status"] == "confirmed" for candidate in all_candidates)

    if fallback_alpr is not None and (not has_confirmed_primary or not all_candidates):
        all_candidates.extend(run_fallback_pipeline(img))

    merged_candidates = expand_checksum_corrections(consolidate_candidates(all_candidates))
    accepted_candidates = [
        candidate
        for candidate in merged_candidates
        if candidate_is_publishable(candidate)
    ][:MAX_RESULTS]

    if not accepted_candidates:
        best_review_candidate = next(
            (
                candidate
                for candidate in merged_candidates
                if candidate_is_visible_review(candidate)
            ),
            None,
        )
        accepted_candidates = [
            prepare_review_candidate_for_display(best_review_candidate)
        ] if best_review_candidate else []

    annotated = annotate_image(img, accepted_candidates) if accepted_candidates else img
    return accepted_candidates, annotated


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    if not primary_pipeline_available() and fallback_alpr is None:
        detail = " | ".join(model_errors) if model_errors else "No model pipeline is available."
        return jsonify({"error": detail}), 500

    try:
        file_bytes = file.read()
        np_arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Image read failed"}), 400

        candidates, annotated = process_image(img)
        plates = [
            {
                "plate": candidate["plate"],
                "display_plate": candidate["display_plate"],
                "confidence": round(candidate["confidence"], 4),
                "ocr_confidence": round(candidate["ocr_confidence"], 4),
                "detector_confidence": round(candidate["detector_confidence"], 4),
                "checksum_match": candidate["checksum_match"],
                "status": candidate["status"],
                "status_label": candidate["status_label"],
                "status_class": "status-confirmed" if candidate["status"] == "confirmed" else "status-review",
                "source": candidate["source"],
                "method": candidate["method"],
                "correction_note": candidate.get("correction_note", ""),
                "display_note": candidate.get("display_note", ""),
                "suppressed_review": candidate.get("suppressed_review", False),
                "is_singapore_plate": candidate.get("is_singapore_plate", True),
            }
            for candidate in candidates
        ]

        return jsonify(
            {
                "success": True,
                "plates": plates,
                "annotated_image": encode_image(annotated),
                "model_warnings": model_errors,
            }
        )

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


def open_browser() -> None:
    webbrowser.open_new("http://127.0.0.1:5000/")


if __name__ == "__main__":
    threading.Timer(1.0, open_browser).start()
    app.run(debug=True, port=5000, use_reloader=False)
