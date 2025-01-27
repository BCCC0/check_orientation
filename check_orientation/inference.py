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


def load_rgb(image_path: Union[Path, str]) -> np.array:
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

class InferenceDataset(Dataset):
    def __init__(self, file_paths: List[Path], transform: albu.Compose) -> None:
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        image_path = self.file_paths[idx]

        image = load_rgb(image_path)

        image = self.transform(image=image)["image"]

        return {"torched_image": tensor_from_rgb_image(image), "image_path": str(image_path)}


def main(input_path, output_path, batch_size=1, num_workers=4):
    # torch.distributed.init_process_group(backend="nccl")
    model = create_model("swsl_resnext50_32x4d")

    with open(args.config_path) as f:
        hparams = yaml.load(f, Loader=yaml.SafeLoader)

    hparams.update(
        {
            "local_rank": args.local_rank,
            "fp16": args.fp16,
        }
    )

    args.output_path.mkdir(parents=True, exist_ok=True)
    hparams["output_path"] = args.output_path

    device = torch.device("cuda")  # pylint: disable=E1101


    corrections: Dict[str, str] = {"model.": ""}
    state_dict = state_dict_from_disk(file_path=args.weight_path, rename_in_layers=corrections)

    model = model.to(device)

    # model = torch.nn.parallel.DistributedDataParallel(
    #     model, device_ids=[args.local_rank], output_device=args.local_rank
    # )

    file_paths = []

    for regexp in ["*.jpg", "*.png", "*.jpeg", "*.JPG"]:
        file_paths += sorted(args.input_path.rglob(regexp))

    # Filter file paths for which we already have predictions
    file_paths = [x for x in file_paths if not (args.output_path / x.parent.name / f"{x.stem}.txt").exists()]

    dataset = InferenceDataset(file_paths, transform=from_dict(hparams["test_aug"]))

    sampler = DistributedSampler(dataset, shuffle=False)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
        sampler=sampler,
    )

    predict(dataloader, model, hparams, device)


def predict(dataloader, model, hparams, device):
    model.eval()

    if hparams["local_rank"] == 0:
        loader = tqdm(dataloader)
    else:
        loader = dataloader

    with torch.no_grad():
        for batch in loader:
            torched_images = batch["torched_image"]  # images that are rescaled and padded

            if hparams["fp16"]:
                torched_images = torched_images.half()

            image_paths = batch["image_path"]

            batch_size = torched_images.shape[0]

            predictions = model(torched_images.to(device))

            for batch_id in range(batch_size):
                file_id = Path(image_paths[batch_id]).stem
                folder_name = Path(image_paths[batch_id]).parent.name

                prob = predictions[batch_id].cpu().numpy().astype(np.float16)

                (hparams["output_path"] / folder_name).mkdir(exist_ok=True, parents=True)

                with open(str(hparams["output_path"] / folder_name / f"{file_id}.txt"), "w") as f:
                    f.write(str(prob.tolist()))


if __name__ == "__main__":
    main()
