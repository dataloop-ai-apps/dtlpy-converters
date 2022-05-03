import dtlpy as dl
from converters import CocoToDataloop

project = dl.projects.get('test-converters-app')
dataset = project.datasets.get('test-coco-converters')
annotation_filepath = '../converters/coco_converters/examples/coco/annotations.json'
images_path = '../converters/coco_converters/examples/images'
to_path = '../converters/coco_converters/examples/dataloop'

conv = CocoToDataloop()
conv.convert_dataset(annotation_filepath=annotation_filepath,
                     to_path=to_path,
                     images_path=images_path,
                     with_upload=True,
                     with_items=True,
                     dataset=dataset)
