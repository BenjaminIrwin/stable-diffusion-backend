from PIL import Image

def create_mask(image_path):
    # Load the image and convert it to RGBA mode
    image = Image.open(image_path).convert("RGBA")
    # Create a new image for the mask, initialized to black
    mask = Image.new("L", image.size, color=0)
    # Iterate over each pixel in the image
    for x in range(image.width):
        for y in range(image.height):
            # Check if the pixel is transparent
            if image.getpixel((x, y))[3] == 0:
                # If it is, set the corresponding pixel in the mask to black
                mask.putpixel((x, y), 0)
            else:
                # Otherwise, set the corresponding pixel in the mask to white
                mask.putpixel((x, y), 255)
    # Save the mask as a PNG file
    mask.save("mask.png")
    return mask


mask = create_mask('layer-inpaint-test.png')
print(locals())

