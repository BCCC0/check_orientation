from collections import namedtuple
from typing import Optional, Dict, Any
from pathlib import Path
import re
import cv2

from timm import create_model as timm_create_model
from torch import nn
from torch.utils import model_zoo

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import albumentations as albu
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
import yaml
from albumentations.core.serialization import from_dict

from torch import nn
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

model = namedtuple("model", ["url", "model"])

models = {
    "swsl_resnext50_32x4d": model(
        model=timm_create_model("swsl_resnext50_32x4d", pretrained=False, num_classes=4),
        url="https://github.com/ternaus/check_orientation/releases/download/v0.0.3/2020-11-16_resnext50_32x4d.zip",
    ),
}


class InferenceDataset(Dataset):
    def __init__(self, file_paths: List[Path]) -> None:
        self.file_paths = file_paths
        self.transform = albu.Compose([albu.Resize(height=224, width=224), albu.Normalize(p=1)], p=1)

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        image_path = self.file_paths[idx]

        image = load_rgb(image_path)
        image = self.transform(image=image)["image"]

        return {"torched_image": tensor_from_rgb_image(image), "image_path": str(image_path)}
        

def load_rgb(image_path) -> np.array:
    """Load RGB image from path.

    Args:
        image_path: path to image
        lib: library used to read an image.
            currently supported `cv2` and `jpeg4py`

    Returns: 3 channel array with RGB image

    """
    if Path(image_path).is_file():
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    raise FileNotFoundError(f"File not found {image_path}")

def tensor_from_rgb_image(image: np.ndarray) -> torch.Tensor:
    image = np.ascontiguousarray(np.transpose(image, (2, 0, 1)))
    return torch.from_numpy(image)


def rename_layers(state_dict: Dict[str, Any], rename_in_layers: Dict[str, Any]) -> Dict[str, Any]:
    result = {}
    for key, value in state_dict.items():
        for key_r, value_r in rename_in_layers.items():
            key = re.sub(key_r, value_r, key)

        result[key] = value

    return result

def create_model(model_name: str, activation: Optional[str] = "softmax") -> nn.Module:
    model = models[model_name].model
    state_dict = model_zoo.load_url(models[model_name].url, progress=True, map_location="cpu")["state_dict"]
    state_dict = rename_layers(state_dict, {"model.": ""})
    model.load_state_dict(state_dict)

    if activation == "softmax":
        return nn.Sequential(model, nn.Softmax(dim=1))

    return model

def main(input_path, img_list, batch_size=2, num_workers=4):
    model = create_model("swsl_resnext50_32x4d").to("cuda")
    input_path = Path(input_path)

    file_paths = [input_path / i for i in img_list]
    # file_paths = []

    # for regexp in ["*.jpg", "*.png", "*.jpeg", "*.JPG"]:
    #     file_paths += sorted(input_path.rglob(regexp))

    # Filter file paths for which we already have predictions
    # file_paths = [x for x in file_paths if not (output_path / x.parent.name / f"{x.stem}.txt").exists()]

    dataset = InferenceDataset(file_paths)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )
    
    return predict(dataloader, model)


def predict(dataloader, model):
    model.eval()
    loader = tqdm(dataloader)
    res = {}
    with torch.no_grad():
        for batch in loader:
            torched_images = batch["torched_image"]  # images that are rescaled and padded

            image_paths = batch["image_path"]

            batch_size = torched_images.shape[0]

            predictions = torch.argmax(model(torched_images.float().to('cuda')), axis = -1).cpu().numpy()

            for batch_id in range(batch_size):
                # prob = predictions[batch_id].cpu().numpy().astype(np.float16)
                res[str(image_paths[batch_id])] = predictions[batch_id]
            
    return res


