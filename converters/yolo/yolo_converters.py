import logging
import json
from pathlib import Path
import dtlpy as dl
import os

from .. import BaseConverter

logger = logging.getLogger(name='dtlpy')


class DataloopToYolo(BaseConverter):
    """
    Annotation Converter
    """

    def on_dataset_start(self, **kwargs):
        return kwargs

    def on_dataset(self, **kwargs):
        """

        :param: local_path: directory to save annotations to
        :param dataset: dl.Dataset
        :param kwargs:
        :return:
        """
        dataset: dl.Dataset = kwargs.get('dataset')
        from_path = kwargs.get('from_path')
        to_path = kwargs.get('to_path')
        dataset.download_annotations(local_path=from_path)
        json_path = Path(from_path).joinpath('json')
        files = list(json_path.rglob('*.json'))
        self.label_to_id_map = dataset.instance_map
        os.makedirs(to_path, exist_ok=True)
        sorted_labels = [k for k, v in sorted(self.label_to_id_map.items(), key=lambda item: item[1])]
        with open(os.path.join(to_path, 'labels.txt'), 'w') as f:
            f.write('\n'.join(sorted_labels))

        for annotation_json_filepath in files:
            with open(annotation_json_filepath, 'r') as f:
                data = json.load(f)
                json_annotations = data.pop('annotations')
                item = dl.Item.from_json(_json=data,
                                         client_api=dl.client_api,
                                         dataset=self.dataset)
                annotations = dl.AnnotationCollection.from_json(_json=json_annotations, item=item)

                self.on_item_end(**self.on_item(**self.on_item_start(item=item,
                                                                     dataset=self.dataset,
                                                                     annotations=annotations,
                                                                     to_path=to_path)))
        return kwargs

    def on_dataset_end(self, **kwargs):
        return kwargs

    def on_item_start(self, **kwargs):
        return kwargs

    def on_item(self, **kwargs):
        item = kwargs.get('item')
        dataset = kwargs.get('dataset')
        annotations = kwargs.get('annotations')
        output_path = kwargs.get('output_path')
        outputs = dict()
        item_yolo_strings = list()
        for i_annotation, annotation in enumerate(annotations.annotations):
            if annotation.type == dl.AnnotationType.BOX:
                outs = {"dataset": dataset,
                        "item": item,
                        "width": item.width,
                        "height": item.height,
                        "annotation": annotation,
                        "annotations": annotations}
                outs = self.on_annotation_end(**self.on_box(**self.on_annotation_start(**outs)))
                item_yolo_strings.append(outs.get('yolo_string'))
                outputs[annotation.id] = outs
        kwargs['outputs'] = outputs
        name, ext = os.path.splitext(item.name)
        output_filename = os.path.join(output_path, item.dir[1:], name + '.txt')
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        with open(output_filename, 'w') as f:
            f.write('\n'.join(item_yolo_strings))
        return kwargs

    def on_box(self, **kwargs):
        """
        Convert from DATALOOP format to YOLO format. Use this as conversion_func param for functions that ask for this param.

        **Prerequisites**: You must be an *owner* or *developer* to use this method.

        :param dtlpy.entities.annotation.Annotation or dict annotation: annotations to convert
        :param dtlpy.entities.item.Item item: item entity
        :return: converted Annotation
        :rtype: tuple
        """
        annotation = kwargs.get('annotation')
        item = kwargs.get('item')
        width = kwargs.get('width')
        height = kwargs.get('height')
        if item.system.get('exif', {}).get('Orientation', 0) in [5, 6, 7, 8]:
            width, height = (item.height, item.width)

        dw = 1.0 / width
        dh = 1.0 / height
        x = (annotation.left + annotation.right) / 2.0
        y = (annotation.top + annotation.bottom) / 2.0
        w = annotation.right - annotation.left
        h = annotation.bottom - annotation.top
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh

        label_id = self.label_to_id_map[annotation.label]
        yolo_string = f'{label_id}, {x}, {y}, {w}, {h}'
        kwargs['yolo_string'] = yolo_string
        return kwargs
