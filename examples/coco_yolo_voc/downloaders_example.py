import dtlpy as dl
from dtlpyconverters.services import DataloopConverters
from dtlpyconverters import coco_converters, yolo_converters, voc_converters

converter = DataloopConverters()
loop = converter._get_event_loop()

# DQL Query is optional
filters = dl.Filters()
query = filters.prepare()

# Convert dataset to COCO
coco_dataset = dl.datasets.get(dataset_id='')
coco_converter = coco_converters.DataloopToCoco(output_annotations_path=r'./output_annotations',
                                                download_annotations=True,
                                                filters=filters,
                                                dataset=coco_dataset)
loop.run_until_complete(coco_converter.convert_dataset())

# Convert dataset to YOLO
yolo_dataset = dl.datasets.get(dataset_id='')
yolo_converter = yolo_converters.DataloopToYolo(output_annotations_path=r'./output_annotations',
                                                download_annotations=True,
                                                filters=filters,
                                                dataset=yolo_dataset)
loop.run_until_complete(yolo_converter.convert_dataset())

# Convert dataset to VOC
voc_dataset = dl.datasets.get(dataset_id='')
voc_converter = voc_converters.DataloopToVoc(output_annotations_path=r'./output_annotations',
                                             download_annotations=True,
                                             filters=filters,
                                             dataset=voc_dataset)
loop.run_until_complete(voc_converter.convert_dataset())
