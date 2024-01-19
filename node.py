from .model import FBCNN
from comfy import model_management
import folder_paths
import comfy
import requests
import os
import torch

class FBCNNNode:
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "image": ("IMAGE",),
                "auto_detect": (["enable", "disable"],),
                "compression_level": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100,
                    "step": 1,
                    "display": "slider"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "fbcnn_node"
    CATEGORY = "image/upscaling"

    def fbcnn_node(self, image, auto_detect, compression_level):
        device = model_management.get_torch_device()
        in_img = image.movedim(-1,-3).to(device)
        
        model_path = folder_paths.get_full_path("FBCNN", "fbcnn_color.pth")
        if model_path is None:
            print(f"Downloading FBCNN model from Github")
            folder = folder_paths.get_folder_paths("FBCNN")
            os.makedirs(folder[0], exist_ok=True)
            url = url = 'https://github.com/jiaxi-jiang/FBCNN/releases/download/v1.0/fbcnn_color.pth'
            r = requests.get(url, allow_redirects=True)
            path = os.path.join(folder[0], "fbcnn_color.pth")
            open(path, 'wb').write(r.content)
            model_path = path
        
        fbcnn = FBCNN()
        fbcnn.load_state_dict(torch.load(model_path), strict=True)
        fbcnn.eval()
        for param in fbcnn.parameters():
            param.requires_grad = False
        fbcnn.to(device)
        
        qf = None if auto_detect == "enable" else torch.tensor([[1-compression_level/100]], dtype=torch.float32, device=device)
        
        tile = 1024
        overlap = 32
        
        oom = True
        while oom:
            try:
                steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap)
                pbar = comfy.utils.ProgressBar(steps)
                s = comfy.utils.tiled_scale(in_img, lambda a: fbcnn.forward(a, qf), tile_x=tile, tile_y=tile, overlap=overlap, upscale_amount=1, pbar=pbar)
                oom = False
            except model_management.OOM_EXCEPTION as e:
                tile //= 2
                if tile < 128:
                    raise e

        fbcnn.cpu()
        s = torch.clamp(s.movedim(-3,-1), min=0, max=1.0)

        return (s,)
        

NODE_CLASS_MAPPINGS = {
    "JPEG artifacts removal FBCNN": FBCNNNode
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "JPEG artifacts removal FBCNN": "JPEG Compression Removal - FBCNN"
}