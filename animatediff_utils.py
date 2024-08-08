import requests
from typing import List
from PIL import Image, ImageEnhance
import os
import base64
import numpy as np

def bright_image_pil(pil_img, factor=0):  # factor > 1은 이미지를 밝게, < 1은 어둡게
    if(factor == 0): return pil_img
    # 밝기 조절
    enhancer = ImageEnhance.Brightness(pil_img)
    img_bright = enhancer.enhance(factor)
    return img_bright

def colored_image_pil(pil_img, factor=0):  # factor > 1은 채도를 증가, < 1은 감소
    if(factor == 0): return pil_img
    # 채도 조절
    enhancer = ImageEnhance.Color(pil_img)
    img_colored = enhancer.enhance(factor)
    return img_colored

def contrast_image_pil(pil_img, factor=0):  # factor > 1은 대비를 증가, < 1은 감소
    if(factor == 0): return pil_img
    # 대비 조절
    enhancer = ImageEnhance.Contrast(pil_img)
    img_contrasted = enhancer.enhance(factor)
    return img_contrasted

def sharp_image_pil(pil_img, factor=0):    # factor > 1은 선명하게, < 1은 blurry하게
    if(factor == 0): return pil_img
    # 선명도 조절
    enhancer = ImageEnhance.Sharpness(pil_img)
    sharp_img = enhancer.enhance(factor)
    return sharp_img


def export_to_mp4(
    sample, output_mp4_path: str = None,
    fps: int = 7, 
    start: int = 0,
    end = None
) -> str:
    import imageio
    image = sample[0].permute(1, 2, 3, 0).cpu().numpy()
    image = image[start:end]
    
    writer = imageio.get_writer(output_mp4_path, fps=fps)
    # Add each frame to the video (twice)
    for img in image:
        frame = (img * 255).astype(np.uint8) 
        writer.append_data(frame)
    for img in image:
        frame = (img * 255).astype(np.uint8) 
        writer.append_data(frame)

    writer.close()


def convert_mp4_to_base64(file_path):
    with open(file_path, 'rb') as mp4_file:
        base64_encoded = base64.b64encode(mp4_file.read())
    return base64_encoded


def encode_image(file: str):
    import base64
    image = file
    if file.startswith('http'):
        response = requests.get(file)
        image = base64.b64encode(response.content).decode('utf-8')
    elif file.endswith('.png') or file.endswith('.jpg') or file.endswith('.jpeg'): #local
        with open(file, "rb") as image_file:
            image = base64.b64encode(image_file.read()).decode('utf-8')
    elif 'base64' in file: # base64
        image = file.split(',')[1]
    else:
        image = file
    return image


def adjust_dimensions(width, height, max_size=640):
    target_area = max_size * max_size
    aspect_ratio = width / height
    
    def find_closest_multiple_of_16(value):
        return int(round(value / 16) * 16)

    # Initial guesses
    new_width = find_closest_multiple_of_16((target_area * aspect_ratio) ** 0.5)
    new_height = find_closest_multiple_of_16((target_area / aspect_ratio) ** 0.5)
    
    if new_width * new_height > target_area:
        new_width = find_closest_multiple_of_16(new_width - 16)
        new_height = find_closest_multiple_of_16(new_height - 16)
    
    return new_width, new_height


def get_controlnet_images(image_paths: str, vae=None, use_simplified_condition_embedding:bool=True, image_max_size=640):
    import torch
    from einops import rearrange, repeat
    from PIL import Image
    import io
    import torchvision.transforms as transforms

    if isinstance(image_paths, str): image_paths = [image_paths]
    print(f"controlnet image paths:")
    for path in image_paths: print(path)
        
    image = Image.open(image_paths[0]).convert("RGB")
    width_prev, height_prev = image.size
    
    width, height = adjust_dimensions(width_prev, height_prev, max_size=image_max_size)
    
    normalize_condition_images = True
    savedir = "output/"

    image_transforms = transforms.Compose([
                    transforms.RandomResizedCrop(
                        (height, width), (1.0, 1.0), 
                        ratio=(width/height, width/height)
                    ),
                    transforms.ToTensor(),
                ])
    if normalize_condition_images == False:
        def image_norm(image):
            image = image.mean(dim=0, keepdim=True).repeat(3,1,1)
            image -= image.min()
            image /= image.max()
            return image
    else: image_norm = lambda x: x

    controlnet_images = [image_norm(image_transforms(Image.open(path).convert("RGB"))) for path in image_paths]

    os.makedirs(os.path.join(savedir, "control_images"), exist_ok=True)
    for i, image in enumerate(controlnet_images):
        Image.fromarray((255. * (image.numpy().transpose(1,2,0))).astype(np.uint8)).save(f"{savedir}/control_images/{i}.png")

    controlnet_images = torch.stack(controlnet_images).unsqueeze(0).cuda()
    controlnet_images = rearrange(controlnet_images, "b f c h w -> b c f h w")

    if use_simplified_condition_embedding and vae is not None:
        num_controlnet_images = controlnet_images.shape[2]
        controlnet_images = rearrange(controlnet_images, "b c f h w -> (b f) c h w")
        controlnet_images = controlnet_images.to(vae.dtype)
        controlnet_images = vae.encode(controlnet_images * 2. - 1.).latent_dist.sample() * 0.18215
        controlnet_images = rearrange(controlnet_images, "(b f) c h w -> b c f h w", f=num_controlnet_images)

    return controlnet_images, width, height