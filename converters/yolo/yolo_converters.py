from pathlib import Path
from PIL import Image
import dtlpy as dl
import numpy as np
import logging
import json
import os

from ..base import BaseConverter

logger = logging.getLogger(name='dtlpy')


class DataloopToYolo(BaseConverter):
    """
    Annotation Converter
    """

    def on_dataset_start(self, **context):
        return context

    def on_dataset(self, **context):
        """

        :param: local_path: directory to save annotations to
        :param dataset: dl.Dataset
        :param context:
        :return:
        """
        # Inputs
        dataset: dl.Dataset = context.get('dataset')
        from_path = context.get('from_path')
        to_path = context.get('to_path')

        #
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
                                                                     to_path=os.path.join(to_path, 'annotations'))))
        return context

    def on_dataset_end(self, **context):
        return context

    def on_item_start(self, **context):
        return context

    def on_item(self, **context):
        item = context.get('item')
        dataset = context.get('dataset')
        annotations = context.get('annotations')
        to_path = context.get('to_path')
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
        context['outputs'] = outputs
        name, ext = os.path.splitext(item.name)
        output_filename = os.path.join(to_path, item.dir[1:], name + '.txt')
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        with open(output_filename, 'w') as f:
            f.write('\n'.join(item_yolo_strings))
        return context

    def on_box(self, **context):
        """
        Convert from DATALOOP format to YOLO format. Use this as conversion_func param for functions that ask for this param.
        **Prerequisites**: You must be an *owner* or *developer* to use this method.
        :param context:
                See below

        :Keyword Arguments:
            * *annotation* (``dl.Annotations``) -- the box annotations to convert
            * *item* (``dl.Item``) -- Item of the annotation
            * *width* (``int``) -- image width
            * *height* (``int``) -- image height
            * *exif* (``dict``) -- exif information (Orientation)

        :return: converted Annotation
        :rtype: tuple
        """
        annotation = context.get('annotation')
        item = context.get('item')
        width = context.get('width')
        height = context.get('height')
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
        yolo_string = f'{label_id} {x} {y} {w} {h}'
        context['yolo_string'] = yolo_string
        return context


class YoloToDataloop:

    def __init__(self):
        ...

    def convert_dataset(self, **context):
        """

        """
        # inputs
        self.annotations_path = context.get('annotations_path')
        self.label_txt_filepath = context.get('label_txt_filepath')
        self.images_path = context.get('images_path')
        self.with_upload = context.get('with_upload')
        self.add_to_recipe = context.get('add_to_recipe', False)
        self.dataset: dl.Dataset = context.get('dataset')

        # read labels and handle recipes
        with open(self.label_txt_filepath, 'r') as f:
            self.id_to_label_map = {i_label: label.strip() for i_label, label in enumerate(f.readlines())}
        if self.add_to_recipe:
            self.dataset.add_labels(label_list=list(self.id_to_label_map.values()))

        # read annotations files and run on items
        files = list(Path(self.annotations_path).rglob('*.txt'))
        for txt_file in files:
            self.on_item(annotation_filepath=str(txt_file))

    def on_item(self, **context):
        annotation_filepath = context.get('annotation_filepath')
        with open(annotation_filepath, 'r') as f:
            lines = f.readlines()

        # find images with the same name (ignore image ext)
        relpath = os.path.relpath(annotation_filepath, self.annotations_path)
        filename, ext = os.path.splitext(relpath)
        image_filepaths = list(Path(os.path.join(self.images_path)).rglob(f'{filename}.*'))
        if len(image_filepaths) != 1:
            assert AssertionError

        # image filepath found
        image_filename = str(image_filepaths[0])
        remote_rel_path = os.path.relpath(image_filename, self.images_path)
        dirname = os.path.dirname(remote_rel_path)
        if self.with_upload:
            # TODO add overwrite as input arg
            item = self.dataset.items.upload(image_filename,
                                             remote_path=f'/{dirname}')
        else:
            try:
                item = self.dataset.items.get(f'/{remote_rel_path}')
            except dl.exceptions.NotFound:
                raise

        if item.width is None:
            width = Image.open(image_filename).size[0]
        else:
            width = item.width
        if item.height is None:
            height = Image.open(image_filename).size[1]
        else:
            height = item.height

        annotation_collection = item.annotations.builder()
        for annotation in lines:
            annotation_collection.annotations.append(self.on_annotation(item=item,
                                                                        annotation=annotation,
                                                                        width=width,
                                                                        height=height))
        item.annotations.upload(annotation_collection)

    def on_annotation(self, **context):
        """
        Convert from COCO format to DATALOOP format. Use this as conversion_func param for functions that ask for this param.

        **Prerequisites**: You must be an *owner* or *developer* to use this method.
        :param context: additional params
        :return: converted Annotation entity
        :rtype: dtlpy.entities.annotation.Annotation
        """
        annotation = context.get('annotation')
        width = context.get('width')
        height = context.get('height')
        item = context.get('item')
        # convert txt line to yolo params as floats
        label_id, x, y, w, h = np.asarray(annotation.strip().split(' ')).astype(float)
        label = self.id_to_label_map.get(int(label_id), f'{label_id}_MISSING')
        x = x * width
        y = y * height
        w = w * width
        h = h * height
        left = x - (w / 2)
        right = x + (w / 2)
        top = y - (h / 2)
        bottom = y + (h / 2)
        ann_def = dl.Box(top=top,
                         bottom=bottom,
                         left=left,
                         right=right,
                         label=label)
        return dl.Annotation.new(annotation_definition=ann_def,
                                 item=item)
