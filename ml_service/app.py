import os
import cv2
import numpy as np
import tensorflow as tf
import traceback
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D

from functional_model import convert_to_functional
from heatmap import generate_heatmap

# Prevent TF logging clutter
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# --------------------
# CONFIG
# --------------------
IMG_SIZE = 224
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
UPLOADS = "uploads"
os.makedirs(UPLOADS, exist_ok=True)

# --------------------
# LOAD & PREPARE MODEL
# --------------------
print("🔹 Loading and converting model...")
raw_model = load_model("saved_model/model.h5")
model = convert_to_functional(raw_model)

# Find the last Conv2D layer for Grad-CAM
last_conv_layer_name = None
for layer in reversed(model.layers):
    if isinstance(layer, Conv2D):
        last_conv_layer_name = layer.name
        break

if not last_conv_layer_name:
    raise RuntimeError("No Conv2D layer found in the model!")

print(f"✅ Model Ready. Target Layer: {last_conv_layer_name}")

# --------------------
# FLASK APP
# --------------------
app = Flask(__name__)
CORS(app)

def preprocess(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Ensure RGB
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def overlay(image_path, heatmap):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    output = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    return output

@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    path = os.path.join(UPLOADS, file.filename)
    file.save(path)

    try:
        # 1. Predict
        input_tensor = preprocess(path)
        preds = model.predict(input_tensor)
        idx = int(np.argmax(preds))
        confidence = float(preds[0][idx])

        # 2. Generate Heatmap
        heatmap = generate_heatmap(input_tensor, model, last_conv_layer_name)

        # 3. Save Result
        result_img = overlay(path, heatmap)
        out_name = f"res_{file.filename}"
        cv2.imwrite(os.path.join(UPLOADS, out_name), result_img)

        return jsonify({
            "prediction": CLASS_NAMES[idx],
            "confidence": round(confidence, 4),
            "image_url": f"http://localhost:5001/files/{out_name}"
        })

    except Exception as e:
        print("❌ ERROR DURING REQUEST:")
        traceback.print_exc() # This prints the error to your terminal
        return jsonify({"error": str(e)}), 500
    
    finally:
        if os.path.exists(path):
            os.remove(path)

@app.route("/files/<filename>")
def files(filename):
    return send_from_directory(UPLOADS, filename)

if __name__ == "__main__":
    app.run(port=5001, debug=False, use_reloader=False)