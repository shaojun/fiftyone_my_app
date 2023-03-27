import os
import shutil

import cv2
import numpy as np

import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F
import crop_background_eb_bicycle_from_local_elenet_dataset


def extract_classwise_instances(samples, output_dir, label_field, ext=".png"):
    print("Extracted object instances...")
    for sample in samples.iter_samples(progress=True):
        img = cv2.imread(sample.filepath)
        img_h, img_w, c = img.shape
        for det in sample.detections.detections:
            if det.label != "Bicycle":
                continue
            [x, y, w, h] = det.bounding_box
            x = int(x * img_w)
            y = int(y * img_h)
            h = int(img_h * h)
            w = int(img_w * w)
            mask_img = img[y:y+h, x:x+w, :]
            # alpha = 255
            # if mask:
            #     alpha = mask.astype(np.uint8)*255
            # alpha = np.expand_dims(alpha, 2)
            # mask_img = np.concatenate((mask_img, alpha), axis=2)

            label = det.label
            label_dir = os.path.join(output_dir, label)
            if not os.path.exists(label_dir):
                os.mkdir(label_dir)
            output_filepath = os.path.join(label_dir, det.id+ext)
            # crop_background_eb_bicycle_from_local_elenet_dataset.resize_image(
            #     mask_img, output_filepath, 224, 224)
            cv2.imwrite(output_filepath, mask_img)


label_field = ""
classes = ["Bicycle"]

dataset = foz.load_zoo_dataset(
    "open-images-v6",
    split="train",
    label_types=["detections"],
    classes=classes,
    max_samples=10000,
    # label_field=label_field,
    dataset_name="shao_test_open_image_bicycle"  # fo.get_default_dataset_name(),
)

#view = dataset.filter_labels(label_field, F("label").is_in(classes))
view = dataset.view()
output_dir = "cropped_open_image_bicycle"
if os.path.exists(output_dir):
    print("removing folder: {}".format(output_dir))
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

extract_classwise_instances(view, output_dir, label_field)
