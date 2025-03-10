import dtlpy as dl
from dtlpyconverters.downloaders import ConvertersDownloader

converter = ConvertersDownloader()

# DQL Query is optional
filters = dl.Filters()

# Convert dataset to COCO
coco_dataset = dl.datasets.get(dataset_id='')
converter.dataloop_to_coco(
    input_annotations_path=r'./input_coco',
    output_annotations_path=r'./output_coco',
    download_annotations=True,
    filters=filters,
    dataset=coco_dataset
)

# Convert dataset to YOLO
yolo_dataset = dl.datasets.get(dataset_id='')
converter.dataloop_to_yolo(
    input_annotations_path=r'./input_yolo',
    output_annotations_path=r'./output_yolo',
    download_annotations=True,
    filters=filters,
    dataset=yolo_dataset
)

# Convert dataset to VOC
voc_dataset = dl.datasets.get(dataset_id='')
converter.dataloop_to_voc(
    input_annotations_path=r'./input_voc',
    output_annotations_path=r'./output_voc',
    download_annotations=True,
    filters=filters,
    dataset=voc_dataset
)
