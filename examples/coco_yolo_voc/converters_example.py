import dtlpy as dl
from dtlpy_converters.services import DataloopConverters

converter = DataloopConverters()

dataset = dl.datasets.get(dataset_id='')

# DQL Query is optional
filters = dl.Filters()
query = filters.prepare()

# Convert dataset to COCO
coco_json_item_id = converter.dataloop_to_coco(dataset=dataset, query=query)
# Convert dataset to yolo
yolo_zip_item_id = converter.dataloop_to_yolo(dataset=dataset, query=query)
# Convert dataset to VOC
voc_zip_item_id = converter.dataloop_to_voc(dataset=dataset, query=query)
