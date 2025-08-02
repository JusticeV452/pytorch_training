import cv2
import numpy as np
import torch


def tensor_to_img(tensor, shape=None, w=None):
    if type(shape) is int and type(w) is int:
        shape = (shape, w)
    h, w = shape if shape is not None else tensor.shape[-2:]
    try:
        np_out = tensor.data.cpu().numpy()
    except:
        np_out = tensor.float().data.cpu().numpy()
    image = (np_out[0].transpose(1, 2, 0)[:h, :w] * 255).astype(dtype='uint8')
    return image


def img_to_tensor(img, device=None):
    if not device or type(device) is str:
        default_name = 'cuda' if torch.cuda.is_available() else 'cpu'
        device_name = device if type(device) is str else default_name
        device = torch.device(device_name)
    return (torch.tensor(img.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)


def write_rgb(file_path, image):
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    if not cv2.imwrite(file_path, image):
        raise Exception(f"Writing to {file_path} failed.")


def resize_image(image, target_shape, preserve_aspect_ratio=False, use_cropping=False, crop_type="random"):
    """
    Resize an image to a specified shape with options for preserving aspect ratio,
    and cropping with various strategies.

    Args:
        image (np.ndarray): Input image (H, W, C).
        target_shape (tuple): Desired output shape (target_height, target_width).
        preserve_aspect_ratio (bool): Whether to preserve the aspect ratio of the image.
        use_cropping (bool): Whether to crop the image to the target shape.
        crop_type (str): Cropping strategy. Options include "center", "random", and combinations
                         of vertical ("top", "middle", "bottom") and horizontal ("left", "center", "right").

    Returns:
        np.ndarray: Resized or cropped image with the specified shape.
    """
    target_height, target_width = target_shape
    original_height, original_width = image.shape[:2]

    if use_cropping:
        # Determine cropping offsets based on crop_type
        if crop_type == "random":
            crop_x = np.random.randint(0, max(1, original_width - target_width + 1))
            crop_y = np.random.randint(0, max(1, original_height - target_height + 1))
        else:
            # Parse crop_type into vertical and horizontal components
            vertical = "middle"
            horizontal = "center"
            for token in crop_type.split():
                if token in ["top", "middle", "bottom"]:
                    vertical = token
                if token in ["left", "center", "right"]:
                    horizontal = token

            # Vertical offset
            if vertical == "top":
                crop_y = 0
            elif vertical == "middle":
                crop_y = max((original_height - target_height) // 2, 0)
            elif vertical == "bottom":
                crop_y = max(original_height - target_height, 0)

            # Horizontal offset
            if horizontal == "left":
                crop_x = 0
            elif horizontal == "center":
                crop_x = max((original_width - target_width) // 2, 0)
            elif horizontal == "right":
                crop_x = max(original_width - target_width, 0)

        # Perform cropping
        cropped_image = image[crop_y:crop_y + target_height, crop_x:crop_x + target_width]
        return cropped_image

    if not preserve_aspect_ratio:
        # Direct resizing to the target shape (no aspect ratio preservation)
        return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)

    # Preserve aspect ratio
    aspect_ratio = original_width / original_height
    target_aspect_ratio = target_width / target_height

    if aspect_ratio > target_aspect_ratio:
        # Image is wider than target: Match width, adjust height
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        # Image is taller than target: Match height, adjust width
        new_height = target_height
        new_width = int(target_height * aspect_ratio)

    # Resize the image while preserving aspect ratio
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Pad the resized image to fit the target shape
    pad_x = (target_width - new_width) // 2
    pad_y = (target_height - new_height) // 2

    padded_image = cv2.copyMakeBorder(
        resized_image,
        top=pad_y, bottom=target_height - (new_height + pad_y),
        left=pad_x, right=target_width - (new_width + pad_x),
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0)  # Padding with black color
    )
    return padded_image
