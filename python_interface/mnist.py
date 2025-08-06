import numpy as np
from tensorflow.keras.datasets import mnist
from PIL import Image

GRID = 14
N = 10000

# Output files
digits_path = "./digits.txt"
labels_path = "./labels.txt"

def resize_and_normalize(img_28x28):
    """Resize to 14x14 and normalize pixel values to [0.0, 1.0]."""
    pil_img = Image.fromarray(img_28x28)
    pil_img = pil_img.resize((GRID, GRID), Image.BILINEAR)
    np_img = np.array(pil_img).astype(np.float32) / 255.0
    return np_img

# Load MNIST
(train_images, train_labels), _ = mnist.load_data()

with open(digits_path, 'w') as digits_file, open(labels_path, 'w') as labels_file:
    for i in range(N):
        image = resize_and_normalize(train_images[i])
        label = train_labels[i]

        # Write digit image
        digits_file.write(f"{GRID} {GRID} {GRID}\n")
        for row in image:
            digits_file.write(' '.join(f"{px:.2f}" for px in row) + '\n')

        # Write label
        labels_file.write(f"{label}\n")

print(f"Saved {N} samples to:")
print(f" - {digits_path}")
print(f" - {labels_path}")
