#predict_model.py
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing import image

MODEL_PATH = "lung_cancer_model.h5"   # <-- Your model file
IMG_SIZE = (224, 224)

# Load model once
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

def preprocess(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def get_stage(prob):
    if prob < 0.60: return "Stage 1 – Suspicious"
    elif prob < 0.75: return "Stage 2 – Early Localized"
    elif prob < 0.90: return "Stage 3 – Advanced"
    return "Stage 4 – High Risk"

def grad_cam(img_path):
    img_array = preprocess(img_path)

    # last conv layer for EfficientNetB0
    last_conv = model.get_layer("top_conv")

    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [last_conv.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, prediction = grad_model(img_array)
        loss = prediction[:, 0]

    grads = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_out = conv_out[0]
    heatmap = conv_out @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap).numpy()

    heatmap = np.maximum(heatmap, 0)
    heatmap /= (heatmap.max() + 1e-8)

    heatmap = cv2.resize(heatmap, (224, 224))

    original = cv2.imread(img_path)
    original = cv2.resize(original, (224, 224))

    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original, 0.6, heatmap_color, 0.4, 0)

    out_path = "static/heatmap.jpg"
    cv2.imwrite(out_path, overlay)

    return out_path

def predict_image(img_path):
    """ Main function used by FastAPI """
    arr = preprocess(img_path)
    pred = model.predict(arr)[0][0]

    cancer_prob = float(pred)
    no_prob = 1 - cancer_prob

    if cancer_prob < 0.5:
        return {
            "result": "NO CANCER",
            "confidence": round(no_prob * 100, 2),
            "severity": 0,
            "stage": "Normal",
            "heatmap": None
        }
    print("MODEL RAW OUTPUT:", cancer_prob)


    stage = get_stage(cancer_prob)
    heatmap_path = grad_cam(img_path)

    return {
        "result": "CANCER",
        "confidence": round(cancer_prob * 100, 2),
        "severity": round(cancer_prob * 100, 2),
        "stage": stage,
        "heatmap": heatmap_path
    }
