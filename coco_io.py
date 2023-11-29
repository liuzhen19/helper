import os
import json
import cv2
import numpy as np
from copy import deepcopy
from pathlib import Path
from glob import glob
from typing import List, Any, Tuple, Dict
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils
Mat = np.ndarray

class Utils():
	@staticmethod
	def cv_imread(filename: str, flags: Any = cv2.IMREAD_UNCHANGED) -> Mat:
		return cv2.imdecode(np.fromfile(filename, np.uint8), flags)

	@staticmethod
	def cv_imwrite(filename: str, img: Mat) -> bool:
		try:
			cv2.imencode(Path(filename).suffix, img)[1].tofile(filename)
			return True
		except Exception:
			return False


class CoCo():
	def __init__(self):
		pass

	def export_to_mask(self, json_file, mask_path, mask_ext, img_ext):
		if not os.path.exists(mask_path):
			os.makedirs(mask_path, exist_ok=True)

		pycoco_inst = COCO(json_file)
		for _, img in pycoco_inst.imgs.items():
			mask = np.zeros((img["height"], img["width"]), np.uint8)
			anns_ids = pycoco_inst.getAnnIds(imgIds=img['id'])
			anns = pycoco_inst.loadAnns(anns_ids)
			for ann in anns:
				mask += pycoco_inst.annToMask(ann)
			mask[ mask > 1 ] = 1
			mask *= 255
			img_basename = img["file_name"]
			mask_basename = img_basename.replace(img_ext, mask_ext)
			mask_filename = os.path.join(mask_path, mask_basename)
			Utils.cv_imwrite(mask_filename, mask)


	def import_from_mask(self, mask_path, mask_ext, img_ext, json_file):
		info_dict = {
				"description": "COCO style dataset",
				"version": "1.0",
				"year": 2023,
				"contributor": "",
				"url": "",
				"date_created": ""
			}
		class_dict = [{"id": 0, "name": "defect", "supercategory": "defect"}]
		coco_data = {
			"info": info_dict,
			"licenses": [],
			"categories": class_dict,
			"images": [],
			"annotations": []
		}

		mask_filename_list = sorted(list(glob(os.path.join(mask_path, f"*{mask_ext}"))))
		annotation_id = 1
		for image_id, mask_filename in enumerate(mask_filename_list, 1):
			img_filename = mask_filename.replace(mask_ext, img_ext)
			img_basename = os.path.basename(img_filename)
			mask = Utils.cv_imread(mask_filename)
			image_h, image_w = mask.shape[0:2]

			image_info = {
				"id": image_id,
				"file_name": img_basename,
				"width": image_w,
				"height": image_h,
				"license": None,
				"date_captured": None
			}
			coco_data["images"].append(image_info)

			# Find contours in the mask
			contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

			for contour in contours:
				poly = contour.reshape(-1).tolist()
				x, y, w, h = cv2.boundingRect(contour)
				area = cv2.contourArea(contour)
				segmentation = [poly]

				annotation = {
					"id": annotation_id,
					"image_id": image_id,
					"category_id": 0,
					"segmentation": segmentation,
					"area": float(area),
					"bbox": [x, y, w, h],
					"iscrowd": 0
				}
				coco_data["annotations"].append(annotation)
				annotation_id += 1

		with open(json_file, "w", encoding="utf-8") as json_fid:
			json.dump(coco_data, json_fid, indent=2, ensure_ascii=False)


def mask_to_coco():
	base_path = "E:/project/Challenge/final/private"
	for stage in ["train", "stage1", "stage2"]:
		for dataset in ["Cable", "Pill", "Wood"]:

			mask_path = os.path.join(base_path, stage, dataset)
			mask_ext  = ".png"
			img_ext	  = ".jpg"
			json_file = os.path.join(mask_path, "annotations.json")
			coco_inst = CoCo()
			coco_inst.import_from_mask(mask_path, mask_ext, img_ext, json_file)
			print(f"Finish {json_file}")


def coco_to_mask():
	base_path = "E:/project/Challenge/final/private"
	tmp_path  = "E:/project/Challenge/final/__tmp__"
	for stage in ["train", "stage1", "stage2"]:
		for dataset in ["Cable", "Pill", "Wood"]:
			json_file = os.path.join(base_path, stage, dataset, "annotations.json")
			mask_path = os.path.join(tmp_path, stage, dataset)
			mask_ext  = ".png"
			img_ext	  = ".jpg"
			coco_inst = CoCo()
			coco_inst.export_to_mask(json_file, mask_path, mask_ext, img_ext)
			print(f"Save to {mask_path}")


if __name__ == "__main__":
	mask_to_coco()
	# coco_to_mask()