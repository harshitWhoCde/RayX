import cv2
import numpy as np
from tensorflow.keras.models import load_model

IMG_SIZE = 224
CLASS_NAMES = ['glioma', 'meningioma',  'notumor' ,'pituitary']

# Load trained model
model = load_model("saved_model/model.h5")

def predict_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)
    class_index = np.argmax(predictions)
    confidence = predictions[0][class_index]

    return CLASS_NAMES[class_index], float(confidence)


if __name__ == "__main__":
    test_image = "dataset\\notumor\Tr-no_0021.jpg"  # change this
    label, conf = predict_image(test_image)
    print(f"Prediction: {label}, Confidence: {conf:.2f} , image : {test_image}" )
