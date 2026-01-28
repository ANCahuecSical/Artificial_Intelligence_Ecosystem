# AI Image Processing and Classification Project

This project is designed to give you hands-on experience working with an image classifier and enhancing your programming skills using AI assistance. The project has two parts, each focused on different aspects of image classification and processing. By the end, you'll have explored fundamental concepts like Grad-CAM, image classification, and creative image filtering.


## Grad-CAM Usage

After running the classifier and getting predictions, you can generate a Grad-CAM visualization to see which areas influenced the model's decision.

1. Run the script:

```bash
python base_classifier.py
```

2. Enter an image path when prompted.
3. After predictions, type `y` at the `Generate Grad-CAM for this image?` prompt to save an overlay image next to the original file (filename appended with `_gradcam.jpg`).

Dependencies for Grad-CAM: `opencv-python`, `matplotlib`, `pillow`, and `tensorflow`.
