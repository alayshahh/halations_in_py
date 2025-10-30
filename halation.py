import argparse
import numpy as np
from PIL import Image, ImageFilter, ImageFile
from pathlib import Path



"""
Uses luminance formula for sRGB 
Returns 2d matrix of luminance per pixel. 
"""
def get_luminance(img_data: np.array) -> np.array:
    red_channel = img_data[:, :, 0]
    green_channel = img_data[:, :, 1]
    blue_channel = img_data[:, :, 2]
    return (0.2126 * red_channel + 0.7152 * green_channel + 0.0722 * blue_channel)

def get_binary_mask_with_blur(luminance: np.array, brightness_threshold: np.uint8, radius: float) -> ImageFile:
    binary_mask = np.where(luminance[:,:] > brightness_threshold, 255, 0).astype(np.uint8)
    return Image.fromarray(binary_mask).filter(ImageFilter.GaussianBlur(radius=radius))

def compose_halation_layer(blurred_mask: Image, r: np.uint8, g: np.uint8, b: np.uint8) -> ImageFile:
    return Image.fromarray(
        np.stack(
            (
                np.full_like(blurred_mask, r, dtype=np.uint8),
                np.full_like(blurred_mask, g, dtype=np.uint8),
                np.full_like(blurred_mask, b, dtype=np.uint8),
                blurred_mask
            ),
            axis=2
        )
    )

def create_halation(img: ImageFile, r: np.uint8, g: np.uint8, b: np.uint8, brightness_threshold: np.uint8, radius: float) -> Image:

    return Image.alpha_composite(
        img.convert("RGBA"), 
        compose_halation_layer(
            get_binary_mask_with_blur(
                get_luminance(
                    np.array(img)
                ),
                brightness_threshold, 
                radius
            ),
            r,
            g,
            b
        )
    ).convert("RGB")


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Add halation to your image")
    parser.add_argument("-p", "--path", type=str, required=True, help="Path to your image")
    parser.add_argument("-s", "--size_blur", type=float, default=250, help="Size of blur radius")
    parser.add_argument("-r", "--red", type=np.uint8, default=255, help="Red color channel for the color tint")
    parser.add_argument("-g", "--green", type=np.uint8, default=100, help="Green color channel for the color tint")
    parser.add_argument("-b", "--blue", type=np.uint8, default=100, help="Blue color channel for the color tint")
    parser.add_argument("-t", "--threshold", type=np.uint8, default=100, help="Brightness threshold used for the halation")

    args = parser.parse_args()
    path = Path(args.path)
    dir = path.parent
    fileExtension = path.suffix.lower()
    if not fileExtension.endswith(".jpeg") and not fileExtension.endswith(".png") and not fileExtension.endswith(".jpg"): 
        raise TypeError("Given file path is not a compatible image. Must be .jpeg, .png or .jpg")
    
    image = Image.open(fp=path)
    new_image = create_halation(image, args.red, args.green, args.blue, args.threshold, args.size_blur)

    new_image_file_name = f'{path.stem}-halation{path.suffix}'

    new_path = path.parent / new_image_file_name

    print(f'Saving image with halation here {new_path}')

    new_image.save(new_path)

