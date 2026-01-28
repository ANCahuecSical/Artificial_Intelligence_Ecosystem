## Grad-CAM Usage (backed up)

After running the classifier and getting predictions, you can generate a Grad-CAM visualization to see which areas influenced the model's decision.

1. Run the script:

```bash
python base_classifier.py
```

2. Enter an image path when prompted.
3. After predictions, type `y` at the `Generate Grad-CAM for this image?` prompt to save an overlay image next to the original file (filename appended with `_gradcam.jpg`).

Dependencies for Grad-CAM: `opencv-python`, `matplotlib`, `pillow`, and `tensorflow`.
