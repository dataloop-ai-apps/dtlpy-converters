import dtlpy as dl
from dtlpyconverters.uploaders import ConvertersUploader

converter = ConvertersUploader()

# Upload COCO local
coco_dataset = dl.datasets.get(dataset_id='')
converter.coco_to_dataloop(dataset=coco_dataset,
                           input_items_path=r'../coco/images',
                           input_annotations_path=r'../coco/coco',
                           coco_json_filename='annotations.json',
                           annotation_options=[dl.AnnotationType.BOX,
                                               dl.AnnotationType.SEGMENTATION],
                           upload_items=True,
                           to_polygon=True)
# Upload YOLO local
yolo_dataset = dl.datasets.get(dataset_id='')
converter.yolo_to_dataloop(dataset=yolo_dataset,
                           input_items_path=r'../yolo/images',
                           input_annotations_path=r'../yolo/yolo/annotations',
                           upload_items=True,
                           add_labels_to_recipe=True,
                           labels_txt_filepath=r'../yolo/yolo/labels.txt'
                           )
# Upload VOC local
voc_dataset = dl.datasets.get(dataset_id='')
converter.voc_to_dataloop(dataset=voc_dataset,
                          input_items_path=r'../voc/images',
                          input_annotations_path=r'../voc/voc/annotations',
                          upload_items=True,
                          add_labels_to_recipe=True
                          )
