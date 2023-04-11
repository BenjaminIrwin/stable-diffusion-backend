# Importing libraries
import base64
from io import BytesIO

import numpy as np
from PIL import Image

# Function which receives a base64 string and returns a base64 string with a hole in the middle
def make_a_hole_in_image(base64_string):
    # Converting base64 string to numpy array
    image = Image.open(BytesIO(base64.b64decode(base64_string)))
    # Convert to transparent
    image = image.convert("RGBA")

    image = np.array(image)

    # Creating a large hole which covers a quarter of the image
    image[int(image.shape[0] / 4):int(image.shape[0] / 4 * 3), int(image.shape[1] / 4):int(image.shape[1] / 4 * 3), :] = 0

    # Converting numpy array to base64 string
    image = Image.fromarray(image)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())

    # Returning base64 string
    return img_str