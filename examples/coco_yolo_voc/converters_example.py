import dtlpy as dl
from dtlpyconverters.services import DataloopConverters

converter = DataloopConverters()

dataset = dl.datasets.get(dataset_id='')

# DQL Query is optional
filters = dl.Filters()
query = filters.prepare()

# Convert dataset to COCO
coco_zip_filepath = converter.dataloop_to_coco(dataset=dataset, query=query, export_locally=True)
# Convert dataset to yolo
yolo_zip_filepath = converter.dataloop_to_yolo(dataset=dataset, query=query, export_locally=True)
# Convert dataset to VOC
voc_zip_filepath = converter.dataloop_to_voc(dataset=dataset, query=query, export_locally=True)
