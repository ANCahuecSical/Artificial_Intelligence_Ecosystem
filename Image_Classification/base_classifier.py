import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from grad_cam import make_gradcam_heatmap, overlay_heatmap

model = MobileNetV2(weights="imagenet")

def classify_image(image_path):
    try:
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        decoded_predictions = decode_predictions(predictions, top=3)[0]

        print("\nTop-3 Predictions for", image_path)
        for i, (_, label, score) in enumerate(decoded_predictions):
            print(f"  {i + 1}: {label} ({score:.2f})")
        try:
            resp = input("Generate Grad-CAM for this image? (y/N): ").strip().lower()
            if resp == 'y':
                heatmap = make_gradcam_heatmap(img_array, model)
                out_path = image_path + "_gradcam.jpg"
                overlay_heatmap(image_path, heatmap, output_path=out_path)
                print(f"Saved Grad-CAM overlay to {out_path}")
        except Exception as e:
            print(f"Could not generate Grad-CAM: {e}")
    except Exception as e:
        print(f"Error processing '{image_path}': {e}")

if __name__ == "__main__":
    print("Image Classifier (type 'exit' to quit)\n")
    while True:
        image_path = input("Enter image filename: ").strip()
        if image_path.lower() == "exit":
            print("Goodbye!")
            break
        classify_image(image_path)
