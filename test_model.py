from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model("Brain_Tumor_Model.h5")

img_path = "dataset/validation/notumor/Te-no_0010.jpg"  # choose any test image

img = image.load_img(img_path, target_size=(224,224))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
img = img / 255.0

pred = model.predict(img)
class_labels = ["glioma", "meningioma", "no_tumor", "pituitary"]
predicted_index = int(np.argmax(pred, axis=1)[0])
confidence = float(np.max(pred)) * 100
result = class_labels[predicted_index]

print(f"Predicted class: {result}, Confidence: {confidence:.2f}%")
