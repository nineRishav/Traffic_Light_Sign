# custom_train.py
from ultralytics import YOLO
# from ultralytics.engine.trainer import DetectionTrainer
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.data.augment import Albumentations
import albumentations as A
import cv2

# --------------- Albumentations pipeline ---------------
bbox_params = A.BboxParams(format="yolo",
                           label_fields=["class_labels"],
                           min_visibility=0.3)

glare_aug = A.Compose([
    A.RandomBrightnessContrast((-0.3, 0.3), (-0.3, 0.3), p=0.4),
    A.RandomSunFlare(flare_roi=(0.1,0.2,0.9,0.5), src_radius=60, p=0.2),
    A.RandomShadow(shadow_roi=(0,0.5,1,1), num_shadows_limit=(1,2),
                   shadow_dimension=5, p=0.2),
    A.GaussNoise(var_limit=(10, 25), mean=0, p=0.2),
    A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.25, p=0.15),
    A.MotionBlur((3,5), p=0.2),
    A.GaussianBlur((3,5), p=0.2),
    A.Affine(translate_percent=0.04, scale=(0.85,1.15),
             rotate=(-5,5), cval=(114,114,114), p=0.6),
    A.Perspective(scale=(0.0005,0.0015), keep_size=True, p=0.3),
    A.CoarseDropout(max_holes=6, max_height=96, max_width=96,
                    fill_value=114, p=0.25),
    A.RandomSizedBBoxSafeCrop(640, 640, erosion_rate=0.2, p=0.5)
], bbox_params=bbox_params)

alb_wrapper = Albumentations(glare_aug)

# --------------- Custom trainer ------------------------
class AlbDetTrainer(DetectionTrainer):
    def set_data(self):
        """call parent, then inject our Albumentations for the train dataset"""
        super().set_data()
        if self.trainset:
            self.trainset.albumentations = alb_wrapper

# --------------- Training overrides --------------------
overrides = dict(
    model="yolov8s.pt",
    data="/DATA1/konda/Traffic_light/traffic-light-mergeDataset-v4/data.yaml",
    epochs=300,
    imgsz=832,
    batch=48,
    device='0',
    lr0=0.001,
    lrf=0.1,
    cos_lr=True,
    warmup_epochs=5,
    patience=40,
    mosaic=0.5,
    mixup=0.05,
    cutmix=0.05,
    copy_paste=0.05,
    hsv_h=0, hsv_s=0.3, hsv_v=0.2,
    degrees=5, shear=1, translate=0.05, scale=0.5, perspective=0.001,
    auto_augment=False,
    augment=True,
    cache=True,
    val=True,
    save_json=True,
    plots=True,
    project="/DATA1/konda/Traffic_light/traffic-light-mergeDataset-v4/models",
    name="trafficlight_aug",
)

if __name__ == "__main__":
    AlbDetTrainer(overrides=overrides).train()
