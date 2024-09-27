import os
import torch
import hydra
import torchvision
import numpy as np
import geopandas as gpd
from pathlib import Path
import tqdm
import rasterio.features
import shapely.geometry
import pycocotools.mask
from segment_anything_hq import (
    SamPredictor as SamPredictor_hq,
    sam_model_registry as sam_model_registry_hq,
)
from segment_anything import SamPredictor, sam_model_registry
import utils
from omegaconf import DictConfig
import logging
from logging.config import fileConfig

import groundingdino.util.inference

# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# assert torch.cuda.is_available()

fileConfig("logging.conf", disable_existing_loggers=False)
logger = logging.getLogger("root")


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:

    # Load Grounding Dino Model
    logger.info(f"Loading grounding dino model from f{cfg.GROUNDING_DINO_CONFIG_PATH}")
    dino_model = groundingdino.util.inference.load_model(
        cfg.GROUNDING_DINO_CONFIG_PATH,
        cfg.GROUNDING_DINO_CHECKPOINT_PATH,
        device=cfg.DEVICE,
    )

    # Load Segment Anything Model (SAM)
    if cfg.USE_SAM_HQ:
        logger.info("Initialize SAM-HQ Predictor")
        sam = sam_model_registry_hq[cfg.SAM_HQ_ENCODER_VERSION](
            checkpoint=cfg.SAM_HQ_CHECKPOINT_PATH
        ).to(device=cfg.DEVICE)
        sam_predictor = SamPredictor_hq(sam)
    else:
        logger.info("Initialize SAM Predictor")
        sam = sam_model_registry[cfg.SAM_ENCODER_VERSION](
            checkpoint=cfg.SAM_CHECKPOINT_PATH
        ).to(device=cfg.DEVICE)
        sam_predictor = SamPredictor(sam)

    # Inladen tilebounds
    df_tilebounds = utils.find_tile_bounds(cfg.path_tifftiles)

    # image id's
    path_instances = Path(cfg.dataDir) / "annotations" / "instances.json"
    imgIds = utils.get_annotation_image_ids(path_instances)

    # dataframe with prediction shapes
    df_pred_shapes = dict(category=[], confidence=[], geometry=[])

    # dataframe with segmentation shapes
    df_seg_shapes = dict(category=[], geometry=[])

    # loop door image id's
    for imgId in tqdm.tqdm(imgIds):

        # path to original image
        imgPath = utils.get_image_path(imgId)

        # Load image
        image_pil, image = utils.load_image(imgPath)

        # annotations
        anns = utils.get_annotations(imgId)

        # tags dino
        tags = utils.get_tags(cfg)

        # Find bounding boxes with grounding dino
        boxes_filt, scores, pred_phrases = utils.get_grounding_output(
            dino_model,
            image,
            tags,
            cfg.DINO_BOX_THRESHOLD,
            cfg.DINO_TEXT_THRESHOLD,
            device=cfg.DEVICE,
        )

        # Resize boxes ?????
        size = image_pil.size
        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        # use NMS to handle overlapped boxes
        boxes_filt = boxes_filt.cpu()
        nms_idx = (
            torchvision.ops.nms(boxes_filt, scores, cfg.IOU_THRESHOLD).numpy().tolist()
        )
        if cfg.DO_IOU_MERGE:
            boxes_filt_clean = boxes_filt[nms_idx]
            pred_phrases_clean = [pred_phrases[idx] for idx in nms_idx]
            # print(f"NMS: before {boxes_filt.shape[0]} boxes, after {boxes_filt_clean.shape[0]} boxes")
        else:
            boxes_filt_clean = boxes_filt
            pred_phrases_clean = pred_phrases

        # Segment objects with SAM
        image_np = np.array(image_pil)
        sam_predictor.set_image(image_np)
        transformed_boxes = sam_predictor.transform.apply_boxes_torch(
            boxes_filt_clean, image_np.shape[:2]
        ).to(cfg.DEVICE)
        masks, _, _ = sam_predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes.to(cfg.DEVICE),
            multimask_output=False,
        )

        # Find tile bounds (X, Y) based on name
        imgName = ".".join(coco.imgs[imgId]["file_name"].split(".")[:-1])
        tile1a = imgName.split("_")[0]
        tile1b = int(imgName.split("_")[-1])
        cellfile = df_tilebounds[
            (df_tilebounds.index == tile1b) & (df_tilebounds.name == tile1a)
        ].copy()
        assert len(cellfile) == 1
        cellfile = cellfile.iloc[0, :].copy()
        xstep = (cellfile.xmax - cellfile.xmin) / image_np.shape[1]
        ystep = (cellfile.ymax - cellfile.ymin) / image_np.shape[0]

        # eventueel nog geometry van cellfile ipv tabel,
        # eventueel test via inladen tiff

        affine = [xstep, 0, cellfile.xmin, 0, -ystep, cellfile.ymax, 0, 0, 1]

        # SAM masks
        assert len(pred_phrases_clean) == len(masks)
        shapes, titles = [], []
        for cat_title, mask in zip(pred_phrases_clean, masks):
            mask = mask.cpu().numpy()
            cat_shapes = rasterio.features.shapes(
                mask.astype(np.uint8), mask=mask, connectivity=4, transform=affine
            )
            for shape, _ in cat_shapes:
                title, confidence = cat_title.replace(")", "").split("(")
                shape = shapely.geometry.shape(shape).simplify(
                    0.01, preserve_topology=True
                )
                if shape.area > 0.01:
                    df_pred_shapes["category"].append(title)
                    df_pred_shapes["confidence"].append(confidence)
                    df_pred_shapes["geometry"].append(shape)

        for catId, catName in zip(catIds, nms):
            for ann in anns[catId]:
                t = coco.imgs[ann["image_id"]]
                if type(ann["segmentation"]["counts"]) == list:
                    rle = pycocotools.mask.frPyObjects(
                        [ann["segmentation"]], t["height"], t["width"]
                    )
                else:
                    rle = [ann["segmentation"]]
                m = pycocotools.mask.decode(rle)[:, :, 0]
                cat_shapes = rasterio.features.shapes(
                    m.astype(np.uint8), mask=m, connectivity=4, transform=affine
                )
                for shape, _ in cat_shapes:
                    title, confidence = cat_title.replace(")", "").split("(")
                    shape = shapely.geometry.shape(shape).simplify(
                        0.01, preserve_topology=True
                    )
                    if shape.area > 0.01:
                        df_seg_shapes["category"].append(catName)
                        df_seg_shapes["geometry"].append(shape)

        # create figure
        utils.plot(
            cfg=cfg, masks=masks, image_np=image_np, catIds=catIds, catNms=catNms
        )

    # create geodataframe with predictions
    df_pred_shapes = gpd.GeoDataFrame(df_pred_shapes, crs="epsg:28992")

    # save file
    if cfg.USE_SAM_HQ:
        out_path = "output/fix_tags_hq.gpkg"
        print(f"save file {out_path}")
        df_pred_shapes.to_file(out_path)
    else:
        out_path = "output/fix_tags.gpkg"
        print(f"save file {out_path}")
        # df_pred_shapes.to_file(out_path)

    # create geodataframe with labels
    df_seg_shapes = gpd.GeoDataFrame(df_seg_shapes, crs="epsg:28992")

    # save file
    out_path = "output/supervisely_tags.gpkg"
    print(f"save file {out_path}")
    # df_seg_shapes.to_file(out_path)


if __name__ == "__main__":
    main()
