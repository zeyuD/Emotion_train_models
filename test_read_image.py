from torchvision.io import read_image
import os
from functions.load_machine_config import load_machine_config
import matplotlib.pyplot as plt

config = load_machine_config()
work_directory = config["data_dir"] + "Emotion/"
actuator = "ur3e/joint/figures"
username = "u0"
emotion = "a"
task = "lw"
posture = "free"
idx = 1


image_file_dir = os.path.join(
            work_directory,
            "segments",
            actuator
        ) + "/" + username + "_" + emotion + "_" + task + "_" + posture + "_" + str(idx) + ".png"

print("Loading image from:", image_file_dir)

# Load image data (150 × 150 RGB)
data = read_image(image_file_dir)
# Why 4 channels? Because read_image loads images with an alpha channel if present.
# If you want to convert it to 3 channels (RGB), you can do so:
if data.shape[0] == 4:
    data = data[:3, :, :]  # Discard alpha channel

# Show image shape
print("Image shape:", data.shape)

# Show the image
plt.imshow(data.permute(1, 2, 0))  # Permute to (H, W, C) for displaying
plt.axis('off')
plt.show()