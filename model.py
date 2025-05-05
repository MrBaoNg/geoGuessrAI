#!/usr/bin/env python3
"""
Script to perform Grounding DINO-based cropping and geolocation prediction using two pretrained branches.
"""
import argparse
import random
import math
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
import timm
import yaml

from groundingdino.util.inference import predict, load_image
from groundingdino.models import build_model
from groundingdino.util.misc import clean_state_dict

# Constants and conversion functions
RAD = math.pi / 180.0

# Convert model output (unit‐vector) to lat/lon
def xyz_to_latlon(xyz: torch.Tensor) -> torch.Tensor:
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    lat = torch.asin(torch.clamp(z, -1, 1)) / RAD
    lon = torch.atan2(y, x) / RAD
    return torch.stack([lat, lon], dim=-1)

# Grounding DINO settings (adjust as needed)
CROP_SIZE = (224, 224)
TEXT_PROMPT = ["object"]
BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.25

# Normalization constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
CROP_MEAN      = IMAGENET_MEAN + IMAGENET_MEAN
CROP_STD       = IMAGENET_STD  + IMAGENET_STD

# Transforms for context crops
class ContextTransform:
    def __init__(self):
        self.augment = transforms.Compose([
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        ])
    def __call__(self, img: Image.Image) -> torch.Tensor:
        w, h = img.size
        if w < 224 or h < 224:
            scale = 224 / min(w, h)
            img = img.resize((int(w*scale), int(h*scale)), Image.BICUBIC)
            w, h = img.size
        left = (w - 224) // 2
        top  = (h - 224) // 2
        img = TF.crop(img, top, left, 224, 224)
        img = self.augment(img)
        return TF.to_tensor(img)

# Transforms for mask crops (background padded to black)
class MaskTransform:
    def __call__(self, img: Image.Image) -> torch.Tensor:
        w, h = img.size
        max_wh = max(w, h)
        # pad to square
        pad_l = (max_wh - w)//2; pad_t = (max_wh - h)//2
        pad_r = max_wh - w - pad_l; pad_b = max_wh - h - pad_t
        img = TF.pad(img, (pad_l, pad_t, pad_r, pad_b), fill=0)
        # ensure >=224
        if max_wh < 224:
            extra = 224 - max_wh
            ep = extra // 2
            img = TF.pad(img, (ep, ep, extra-ep, extra-ep), fill=0)
        # center-crop if larger
        w2, h2 = img.size
        if w2 > 224 or h2 > 224:
            left = (w2 - 224)//2; top = (h2 - 224)//2
            img = TF.crop(img, top, left, 224, 224)
        return TF.to_tensor(img)

# Transforms for full-image branch
class FullImageTransform:
    def __init__(self):
        self.augment = transforms.Compose([
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        ])
    def __call__(self, img: Image.Image) -> torch.Tensor:
        w, h = img.size
        if w < 224 or h < 224:
            scale = max(224/w, 224/h)
            img = img.resize((int(w*scale), int(h*scale)), Image.BICUBIC)
            w, h = img.size
        max_dx = int(0.1 * (w - 224))
        max_dy = int(0.1 * (h - 224))
        cx = (w - 224)//2 + random.randint(-max_dx, max_dx)
        cy = (h - 224)//2 + random.randint(-max_dy, max_dy)
        cx = max(0, min(cx, w - 224))
        cy = max(0, min(cy, h - 224))
        img = TF.crop(img, cy, cx, 224, 224)
        img = self.augment(img)
        return TF.to_tensor(img)

# Instantiate shared transforms and normalizers
context_tf  = ContextTransform()
mask_tf     = MaskTransform()
full_tf     = FullImageTransform()
normalize_crop = transforms.Normalize(CROP_MEAN, CROP_STD)
normalize_full = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

# Combine context + mask channels
def crop_transform(ctx: Image.Image, mask: Image.Image) -> torch.Tensor:
    return torch.cat([context_tf(ctx), mask_tf(mask)], dim=0)

# Load a Grounding DINO model from config & checkpoint
def load_dino(config_path: Path, checkpoint_path: Path, device: torch.device):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    model = build_model(cfg)
    ckpt = torch.load(checkpoint_path, map_location=device)
    sd = clean_state_dict(ckpt.get('model', ckpt))
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model

class GeoProcessor:
    def __init__(self,
                 dino_config, dino_ckpt,
                 crop_model_path, full_model_path,
                 crop_model_name, full_model_name,
                 device):
        self.device = device
        self.dino   = load_dino(dino_config, dino_ckpt, device)
        # load crop branch
        self.crop_model = timm.create_model(
            crop_model_name, pretrained=False, num_classes=3, in_chans=6
        ).to(device)
        ck = torch.load(crop_model_path, map_location=device)
        sd = ck.get('model_state_dict', ck)
        self.crop_model.load_state_dict(sd, strict=False)
        self.crop_model.eval()
        # load full branch
        self.full_model = timm.create_model(
            full_model_name, pretrained=False, num_classes=3, in_chans=3
        ).to(device)
        ck2 = torch.load(full_model_path, map_location=device)
        sd2 = ck2.get('model_state_dict', ck2)
        self.full_model.load_state_dict(sd2, strict=False)
        self.full_model.eval()

    # generate context+mask crops via DINO
    def _crops(self, img: Image.Image):
        w, h = img.size
        boxes, _, _ = predict(
            self.dino,
            load_image(img)[1],
            TEXT_PROMPT,
            BOX_THRESHOLD,
            TEXT_THRESHOLD,
            self.device
        )
        out = []
        for (i, b) in enumerate(boxes):
            x1, y1, x2, y2 = [int(v*c) for v,c in zip(b, (w,h,w,h))]
            if x2 <= x1 or y2 <= y1:
                continue
            sq = img.crop((x1, y1, x2, y2))
            ctx = sq.resize(CROP_SIZE, Image.BILINEAR)
            bg = Image.new('RGB', sq.size, (0,0,0))
            bg.paste(sq, (0,0))  # black background
            mask = bg.resize(CROP_SIZE, Image.NEAREST)
            out.append((ctx, mask))
        return out

    # jittered center 224x224 full image crop
    def full_crop(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w < 224 or h < 224:
            scale = max(224/w, 224/h)
            img = img.resize((int(w*scale), int(h*scale)), Image.BICUBIC)
            w, h = img.size
        max_dx = int(0.1*(w-224)); max_dy = int(0.1*(h-224))
        cx = (w-224)//2 + random.randint(-max_dx, max_dx)
        cy = (h-224)//2 + random.randint(-max_dy, max_dy)
        cx = max(0, min(cx, w-224)); cy = max(0, min(cy, h-224))
        return TF.crop(img, cy, cx, 224, 224)

    # process a single image path and print lat/lon predictions
    def process_image(self, img_path: Path):
        img = Image.open(img_path).convert('RGB')
        crops = self._crops(img)
        full  = self.full_crop(img)

        # crop branch predictions
        crop_preds = []
        for ctx, mask in crops:
            inp = crop_transform(ctx, mask).unsqueeze(0)
            inp = normalize_crop(inp.squeeze(0)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out = self.crop_model(inp)
                pred = F.normalize(out, dim=-1)
            crop_preds.append(pred)
        cp = torch.cat(crop_preds).mean(0, keepdim=True) if crop_preds else None

        # full branch prediction
        inp_f = full_tf(full).unsqueeze(0)
        inp_f = normalize_full(inp_f.squeeze(0)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out_f = self.full_model(inp_f)
            fp = F.normalize(out_f, dim=-1)

        # ensemble and convert to lat/lon
        final = (0.5*fp + 0.5*cp) if cp is not None else fp
        latlon_ens = xyz_to_latlon(final).cpu().squeeze(0).tolist()

        print(f"Image: {img_path}")
        if cp is not None:
            ll_crop_only = xyz_to_latlon(cp).cpu().squeeze(0).tolist()
            print(f"  Crop-only pred:    {ll_crop_only}")
        full_ll = xyz_to_latlon(fp).cpu().squeeze(0).tolist()
        print(f"  Full-only pred:    {full_ll}")
        print(f"  Ensembled pred:    {latlon_ens}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--crop_model_path',   type=Path, default='geoLocator/checkpoints/smaller_crop_model.pth')
    parser.add_argument('--full_model_path',   type=Path, default='geoLocator/checkpoints/full_model.pth')
    parser.add_argument('--crop_model_name',   type=str, default='resnest50d')
    parser.add_argument('--full_model_name',   type=str, default='convnextv2_tiny')
    parser.add_argument('--dino_config',       type=Path, required=True)
    parser.add_argument('--dino_checkpoint',   type=Path, required=True)
    parser.add_argument('--images',   type=Path, required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    proc = GeoProcessor(
        args.dino_config, args.dino_checkpoint,
        args.crop_model_path, args.full_model_path,
        args.crop_model_name, args.full_model_name,
        device
    )
    
    proc.process_image(args.images)

if __name__ == '__main__':
    main()
