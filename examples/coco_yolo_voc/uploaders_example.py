import dtlpy as dl
from dtlpyconverters.coco import CocoToDataloop
from dtlpyconverters.yolo import YoloToDataloop
from dtlpyconverters.voc import VocToDataloop


# Upload COCO local
coco_dataset = dl.datasets.get(dataset_id='')
converter = CocoToDataloop(
    dataset=coco_dataset,
    input_items_path=r'../coco/images',
    input_annotations_path=r'../coco/coco',
    upload_items=True,
)
converter.convert(
    coco_json_filename='annotations.json',
    annotation_options=[
        dl.AnnotationType.BOX,
        dl.AnnotationType.SEGMENTATION
    ],
    to_polygon=True
)

# Upload YOLO local
yolo_dataset = dl.datasets.get(dataset_id='')
converter = YoloToDataloop(
    dataset=yolo_dataset,
    input_items_path=r'../yolo/input',
    input_annotations_path=r'../yolo/yolo/annotations',
    upload_items=True,
    add_labels_to_recipe=True,
)
converter.convert(
    labels_txt_filepath=r'../yolo/yolo/labels.txt'
)

# Upload VOC local
voc_dataset = dl.datasets.get(dataset_id='')
converter = VocToDataloop(
    dataset=voc_dataset,
    input_items_path=r'../voc/images',
    input_annotations_path=r'../voc/voc/annotations',
    upload_items=True,
    add_labels_to_recipe=True
)
converter.convert()
