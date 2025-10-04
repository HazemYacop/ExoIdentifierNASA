"""Utility helpers to run the exoplanet classifier."""
from __future__ import annotations

import io
import math
import threading
from pathlib import Path
from typing import Tuple

import numpy as np
from django.core.exceptions import ImproperlyConfigured
from PIL import Image

MODEL_RELATIVE_PATH = Path(__file__).resolve().parent / "model_artifacts" / "exoplanet_identifier.h5"

_model = None
_model_lock = threading.Lock()
_input_shape: Tuple[int, int] | None = None


def _load_tensorflow_model():
    """Return a cached TensorFlow model instance along with expected input shape."""
    global _model, _input_shape
    if _model is not None and _input_shape is not None:
        return _model, _input_shape

    with _model_lock:
        if _model is None or _input_shape is None:
            try:
                from tensorflow import keras  # type: ignore
            except ModuleNotFoundError as exc:  # pragma: no cover
                raise ImproperlyConfigured(
                    "TensorFlow is required to run predictions. Install tensorflow-cpu and redeploy."
                ) from exc

            if not MODEL_RELATIVE_PATH.exists():
                raise ImproperlyConfigured(f"Model weights not found at {MODEL_RELATIVE_PATH}")

            model = keras.models.load_model(MODEL_RELATIVE_PATH)
            shape = getattr(model, "input_shape", None)
            if not shape or len(shape) < 3:
                raise ImproperlyConfigured("Unexpected model input shape; expected (batch, steps, channels).")

            steps = shape[1]
            channels = shape[2]
            if channels != 1:
                raise ImproperlyConfigured("This loader expects single-channel inputs.")

            if steps is None:
                raise ImproperlyConfigured("Model input length is undefined; cannot infer resize factor.")

            side = int(math.sqrt(steps))
            if side * side != steps:
                raise ImproperlyConfigured(
                    "Cannot infer square dimensions from the saved model. Please update the pre-processing logic."
                )

            _model = model
            _input_shape = (steps, side)

    if _model is None or _input_shape is None:
        raise ImproperlyConfigured("Model failed to load correctly.")

    return _model, _input_shape


def _prepare_image(image_bytes: bytes, steps: int, side: int) -> np.ndarray:
    """Convert raw bytes into the tensor shape expected by the network."""
    with Image.open(io.BytesIO(image_bytes)) as img:
        grayscale = img.convert("L")
        resized = grayscale.resize((side, side))
        values = np.asarray(resized, dtype="float32") / 255.0
        sequence = values.flatten().reshape(1, steps, 1)
        return sequence


def predict_proba(image_bytes: bytes) -> Tuple[float, str]:
    """Return the probability and label for the supplied image."""
    model, (steps, side) = _load_tensorflow_model()
    input_tensor = _prepare_image(image_bytes, steps, side)
    prediction = model.predict(input_tensor, verbose=0)
    probability = float(prediction[0][0])
    label = "exoplanet" if probability >= 0.5 else "not-exoplanet"
    return probability, label
