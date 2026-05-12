from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model("Brain_Tumor_Model.h5")

# ✅ Must match class_indices printed during training (alphabetical)
# Keras assigns: glioma=0, meningioma=1, notumor=2, pituitary=3
brain_tumor_labels = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

img_path = "dataset/validation/notumor/Te-no_0010.jpg"

img = image.load_img(img_path, target_size=(224, 224))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = img / 255.0

pred = model.predict(img)
predicted_index = int(np.argmax(pred, axis=1)[0])
confidence = float(np.max(pred)) * 100
result = brain_tumor_labels[predicted_index]

print(f"Raw predictions: {pred}")
print(f"Predicted: {result} ({confidence:.2f}% confidence)")
