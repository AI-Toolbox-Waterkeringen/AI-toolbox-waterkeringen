import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union

import geopandas as gpd
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from groundingdino.util.utils import get_phrases_from_posmap
from omegaconf import DictConfig
from PIL import Image
from pycocotools.coco import COCO


def load_image(image_path: Union[str, Path]) -> Tuple[Image.Image, torch.Tensor]:
    """
    Load an image from a specified file path and apply transformations suitable for models like Grounding Dino or SAM.

    Parameters:
    - image_path (str): The file path to the image that needs to be loaded.

    Returns:
    - image_pil (PIL.Image.Image): The original image in PIL format.
    - transformed_image (torch.Tensor): The transformed image as a tensor, suitable for model input.
    """
    # Load and convert image to RGB
    try:
        image_pil = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"The image at {image_path} dofrom groundingdino.util.utils import get_phrases_from_posmapes not exist."
        )
    except Exception as e:
        raise Exception(f"An error occurred while loading the image: {e}")

    # Define transformations
    transform = T.Compose(
        [
            T.RandomResizedCrop(
                800, scale=(0.5, 1.0)
            ),  # Adjusted for variable resizing
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Apply transformations
    transformed_image = transform(image_pil)

    return image_pil, transformed_image


# def get_tags(cfg: DictConfig) -> str:
#     """
#     Get tags from config file and updates to the required format.
#     """
#     # Tags
#     if cfg.use_fixed_tags:
#         tags = ",".join(cfg.fixed_tags)
#     else:
#         # Find tags with RAM
#         ram_model = ram_model.to(cfg.DEVICE)
#         raw_image = image_pil.resize((384, 384))
#         raw_image = transform(raw_image).unsqueeze(0).to(cfg.DEVICE)
#         res = inference_ram(raw_image, ram_model)
#         tags = res[0].replace(" |", ",")
#     return tags


def get_category_names(coco: COCO) -> List[str]:
    """
    Retrieve category names from a COCO dataset.

    Parameters:
    - coco (COCO): The COCO dataset instance.

    Returns:
    - List[str]: A list of category names.
    """
    cats = coco.loadCats(coco.getCatIds())
    catNms = [cat["name"] for cat in cats]
    return catNms


def get_category_ids(coco: COCO) -> List[int]:
    """
    Retrieve category IDs from a COCO dataset.

    Parameters:
    - coco (COCO): The COCO dataset instance.

    Returns:
    - List[int]: A list of category IDs.
    """
    # cats = coco.loadCats(coco.getCatIds())
    catNms = get_category_names(coco)
    catIds = coco.getCatIds(catNms=catNms)
    return catIds


def get_annotations(coco: COCO, imgId: int) -> Dict[int, list]:
    """
    Retrieve annotations for a given image ID from a COCO dataset.

    Parameters:
    - coco (COCO): The COCO dataset instance.
    - imgId (int): The image ID for which annotations are to be retrieved.

    Returns:
    - Dict[int, list]: A dictionary where keys are category IDs and values are lists of annotations.
    """
    # COCO format image path and annotation
    img = coco.loadImgs(imgId)[0]
    catIds = get_category_ids(coco)
    anns = {}
    for catId in catIds:
        anns[catId] = coco.loadAnns(
            coco.getAnnIds(imgIds=img["id"], catIds=[catId], iscrowd=None)
        )
    return anns


def get_image_path(cfg: DictConfig, coco: COCO, imgId: int) -> Path:
    """
    Retrieve the file path for a given image ID from a COCO dataset.

    Parameters:
    - cfg (DictConfig): config
    - coco (COCO): The COCO dataset instance.
    - imgId (int): The image ID for which the path is to be retrieved.

    Returns:
    - Path: The file path to the image.
    """
    # COCO format image path and annotation
    img = coco.loadImgs(imgId)[0]
    imgPath = Path(cfg.dataDir).joinpath("images").joinpath(img["file_name"])
    return imgPath


def get_annotation_image_ids(path_instances: Union[str, Path]) -> List[int]:
    """
    Returns a list of image IDs from Supervisely annotations.

    Patameters
    - path_instance (str): path to annotation instances.


    Returns:
    - List[int]: A list of image IDs from the annotations.
    """
    coco_instance = COCO(path_instances)
    # cats = coco_instance.loadCats(coco_instance.getCatIds())
    # nms = [cat["name"] for cat in cats]
    # catIds = coco_instance.getCatIds(catNms=nms)
    imgIds = coco_instance.getImgIds()
    return imgIds


def find_tile_bounds(
    root_tilepath: Union[str, Path], concat: bool = True
) -> pd.DataFrame:
    """
    Returns a dataframe with cell boundaries of labeled images
    """
    cell_files = []
    for p in Path(root_tilepath).iterdir():
        if p.is_dir():
            cell_files += find_tile_bounds(p, concat=False)
        elif p.is_file() and p.suffix == ".gpkg" and "cells_intersect" in p.stem:
            df = gpd.read_file(p)
            df["name"] = p.stem.replace("_cells_intersects", "")
            cell_files.append(df.copy())
    if concat:
        cell_files = pd.concat(cell_files)
    return cell_files


def get_grounding_output(
    model: torch.nn.Module,
    image: torch.Tensor,
    caption: str,
    box_threshold: float,
    text_threshold: float,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Process an image and caption through a model to generate grounded outputs,
    including filtered bounding boxes and corresponding text phrases.

    Parameters:
    - model (torch.nn.Module): The model to process the input data.
    - image (torch.Tensor): The image tensor.
    - caption (str): The caption string related to the image.
    - box_threshold (float): The threshold value to filter the bounding boxes based on confidence scores.
    - text_threshold (float): The threshold value to filter the text based on logits.
    - device (str, optional): The device type, 'cpu' or 'cuda', where the computation will take place. Defaults to 'cpu'.

    Returns:
    - tuple:
        - filtered_boxes (torch.Tensor): The filtered bounding boxes.
        - scores (torch.Tensor): The confidence scores of the phrases.
        - pred_phrases (list of str): The predicted phrases associated with the bounding boxes.
    """
    # Prepare caption
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption += "."

    # Move model and image to the specified device
    model = model.to(device)
    image = image.to(device)

    # Generate predictions
    try:
        with torch.no_grad():
            outputs = model(
                image.unsqueeze(0), captions=[caption]
            )  # Ensure image is 4D
        logits = outputs["pred_logits"].sigmoid()[0]  # (num_queries, num_classes)
        boxes = outputs["pred_boxes"][0]  # (num_queries, 4)

        # Filter outputs based on thresholds
        max_logits = logits.max(dim=1)[0]
        filt_mask = max_logits > box_threshold
        logits_filt = logits[filt_mask]
        boxes_filt = boxes[filt_mask]

        # Prepare phrases and scores
        tokenizer = model.tokenizer
        tokenized = tokenizer(caption)
        pred_phrases, scores = [], []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(
                logit > text_threshold, tokenized, tokenizer
            )
            pred_phrases.append(f"{pred_phrase} ({logit.max().item():.4f})")
            scores.append(logit.max().item())

        return boxes_filt, torch.tensor(scores), pred_phrases
    except Exception as e:
        raise Exception(f"An error occurred during model prediction: {e}")


def show_mask(
    mask: np.ndarray, ax: matplotlib.axes.Axes, random_color: bool = False
) -> None:
    """
    Display a mask over an axis (ax) with an option to use a random color or a default color.

    Parameters:
    - mask (numpy.ndarray): The mask array to display. Expected shape is (height, width).
    - ax (matplotlib.axes.Axes): The matplotlib axis on which to display the mask.
    - random_color (bool, optional): If True, displays the mask in a random color. Defaults to False, using a deep sky blue color.

    """
    if random_color:
        color = np.random.rand(3)  # Generates three random floats between 0 and 1
        color = np.append(color, 0.6)  # Add alpha for transparency
    else:
        color = np.array(
            [30 / 255, 144 / 255, 255 / 255, 0.6]
        )  # Deep sky blue with transparency

    h, w = mask.shape
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    return None


def show_box(box: Iterable[float], ax: matplotlib.axes.Axes, label: str) -> None:
    """
    Draw a rectangle (box) on an axis (ax) based on given coordinates and add a label to it.

    Parameters:
    - box (iterable): Coordinates of the box as (x_min, y_min, x_max, y_max).
    - ax (matplotlib.axes.Axes): The matplotlib axis on which to draw the box.
    - label (str): The text label to display at the top-left corner of the box.

    """
    x0, y0 = box[0], box[1]
    w, h = box[2] - x0, box[3] - y0
    rect = plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor="none", lw=2)
    ax.add_patch(rect)
    ax.text(
        x0,
        y0,
        label,
        verticalalignment="top",
        color="white",
        fontsize=8,
        bbox={"facecolor": "green", "alpha": 0.5},
    )
    return None


def plot(
    cfg: DictConfig,
    masks: List[torch.Tensor],
    image_np: np.ndarray,
    catIds: List[int],
    catNms: List[str],
    anns: dict,
    boxes_filt_clean: List[torch.Tensor],
    pred_phrases_clean: List[str],
    imgPath: Path,
    tags: str,
    plot_dir: Union[str, Path],
) -> None:
    """
    Generates a plot with two subplots: one showing annotations in COCO format and another showing results
    from a specific model like RAM-DINO-SAM with filtered boxes and phrases.

    Parameters:
    - masks (list of torch.Tensor): List of masks to display in the RAM-DINO-SAM image.
    - image_np (numpy.ndarray): The image data as a numpy array.
    - catIds (list): Category IDs from COCO annotations.
    - catNms (list): Category names corresponding to catIds.
    - anns (dict): Annotations dictionary keyed by category ID.
    - boxes_filt_clean (list of torch.Tensor): Filtered bounding boxes.
    - pred_phrases_clean (list): Corresponding labels for the filtered boxes.
    - imgPath (Path): Path object for the image to be saved.
    - tags (str): Tags to be displayed in the subplot titles.
    """
    # Setup figure and axes
    fig, axs = plt.subplots(1, 2, figsize=(20, 10), dpi=100, squeeze=False)

    # First subplot: COCO formatted annotations
    ax = axs[0, 0]
    ax.imshow(image_np)
    for catId, catName in zip(catIds, catNms):
        if catId in anns:
            for ann in anns[catId]:
                ax.add_patch(
                    plt.Rectangle(
                        (ann["bbox"][0], ann["bbox"][1]),
                        ann["bbox"][2],
                        ann["bbox"][3],
                        edgecolor="red",
                        facecolor="none",
                        lw=2,
                    )
                )
                ax.text(
                    ann["bbox"][0],
                    ann["bbox"][1] - 10,
                    catName,
                    color="white",
                    fontsize=10,
                    bbox={"facecolor": "red", "alpha": 0.5},
                )

    # Second subplot: RAM-DINO-SAM image
    ax = axs[0, 1]
    ax.imshow(image_np)
    for mask in masks:
        show_mask(mask.cpu().numpy(), ax, random_color=True)
    for box, label in zip(boxes_filt_clean, pred_phrases_clean):
        show_box(box.numpy(), ax, label)

    # Titles and axis settings
    axs[0, 0].set_title("Supervisely Annotations")
    axs[0, 1].set_title(f"RAM-DINO-SAM, tags: {tags}", wrap=True)
    axs[0, 0].axis("off")
    axs[0, 1].axis("off")

    # Save figure
    if not plot_dir.exists():
        plot_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_dir.joinpath(imgPath.name), bbox_inches="tight")

    fig.close()


def save_mask_data(
    output_dir: Union[str, Path],
    tags_chinese: str,
    mask_list: List[torch.Tensor],
    box_list: List[torch.Tensor],
    label_list: List[str],
) -> None:
    """
    Save image and JSON data for masks, boxes, and labels.

    Parameters:
    - output_dir (str): Directory to save the output files.
    - tags_chinese (str): Chinese tags to include in the JSON data.
    - mask_list (torch.Tensor): List of mask tensors.
    - box_list (List[torch.Tensor]): List of box tensors.
    - label_list (List[str]): List of label strings.
    """
    value = 0  # 0 for background
    mask_img = torch.zeros(mask_list.shape[-2:])
    for idx, mask in enumerate(mask_list):
        mask_img[mask.cpu().numpy()[0]] = value + idx + 1
    plt.figure(figsize=(10, 10))
    plt.imshow(mask_img.numpy())
    plt.axis("off")
    plt.savefig(
        os.path.join(output_dir, "mask.jpg"),
        bbox_inches="tight",
        dpi=300,
        pad_inches=0.0,
    )
    json_data = {
        "tags_chinese": tags_chinese,
        "mask": [{"value": value, "label": "background"}],
    }
    for label, box in zip(label_list, box_list):
        value += 1
        name, logit = label.split("(")
        logit = logit[:-1]  # the last is ')'
        json_data["mask"].append(
            {
                "value": value,
                "label": name,
                "logit": float(logit),
                "box": box.numpy().tolist(),
            }
        )
    with open(os.path.join(output_dir, "label.json"), "w") as f:
        json.dump(json_data, f)
