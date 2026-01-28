from PIL import Image, ImageFilter, ImageEnhance
import matplotlib.pyplot as plt
import os
import numpy as np

def apply_blur_filter(image_path, output_path="blurred_image.png"):
    try:
        img = Image.open(image_path)
        img_resized = img.resize((128, 128))
        img_blurred = img_resized.filter(ImageFilter.GaussianBlur(radius=2))

        plt.imshow(img_blurred)
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Processed image saved as '{output_path}'.")

    except Exception as e:
        print(f"Error processing image: {e}")


def apply_bitdepth_filter(image_path, bits=4, output_path="bitdepth_image.png"):
    """Reduce color bit depth per channel to `bits` (1-8).

    Uses uniform quantization per channel and saves the result to `output_path`.
    """
    try:
        if bits < 1 or bits > 8:
            raise ValueError("bits must be between 1 and 8")

        img = Image.open(image_path).convert('RGB')
        img_resized = img.resize((128, 128))
        arr = np.array(img_resized, dtype=np.float32)

        levels = 2 ** bits
        # scale to [0, levels), quantize, then rescale to [0,255]
        scaled = np.floor(arr / 256.0 * levels)
        quantized = (scaled * (255.0 / (levels - 1))).clip(0, 255).astype(np.uint8)

        img_out = Image.fromarray(quantized)

        plt.imshow(img_out)
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Processed bit-depth image saved as '{output_path}'.")

    except Exception as e:
        print(f"Error processing image: {e}")


def apply_cool_filter(image_path, intensity=0.5, output_path="cool_image.png"):
    """Apply a cool-tone color filter.

    intensity: 0.0 (no change) to 1.0 (strong effect)
    """
    try:
        intensity = float(intensity)
        if intensity < 0:
            intensity = 0.0
        if intensity > 1:
            intensity = 1.0

        img = Image.open(image_path).convert('RGB')
        img_resized = img.resize((128, 128))

        # blue overlay to shift color temperature
        blue_overlay = Image.new('RGB', img_resized.size, (10, 80, 220))
        blended = Image.blend(img_resized, blue_overlay, alpha=intensity * 0.6)

        # increase color saturation and contrast slightly
        enhancer_color = ImageEnhance.Color(blended)
        saturated = enhancer_color.enhance(1.0 + intensity * 0.6)
        enhancer_contrast = ImageEnhance.Contrast(saturated)
        final = enhancer_contrast.enhance(1.0 + intensity * 0.15)

        plt.imshow(final)
        plt.axis('off')
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Processed cool-tone image saved as '{output_path}'.")

    except Exception as e:
        print(f"Error processing image: {e}")


if __name__ == "__main__":
    print("Image Blur Processor (type 'exit' to quit)\n")
    while True:
        image_path = input("Enter image filename (or 'exit' to quit): ").strip()
        if image_path.lower() == 'exit':
            print("Goodbye!")
            break
        if not os.path.isfile(image_path):
            print(f"File not found: {image_path}")
            continue
        # derive output filename
        base, ext = os.path.splitext(image_path)
        output_file = f"{base}_blurred{ext}"
        apply_blur_filter(image_path, output_file)