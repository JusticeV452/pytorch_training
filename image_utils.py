import cv2
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
