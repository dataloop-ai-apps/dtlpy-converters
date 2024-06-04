from pathlib import Path
from PIL import Image
import dtlpy as dl
import numpy as np
import logging
import time
import json
import os
import cv2

from ..base import BaseExportConverter, BaseImportConverter

logger = logging.getLogger(name='dtlpy')


class YoloToDataloop(BaseImportConverter):
    def __init__(self,
                 dataset: dl.Dataset,
                 input_annotations_path,
                 output_annotations_path=None,
                 input_items_path=None,
                 upload_items=False,
                 add_labels_to_recipe=True,
                 concurrency=6,
                 return_error_filepath=False,
                 ):
        # global vars
        super(YoloToDataloop, self).__init__(
            dataset=dataset,
            output_annotations_path=output_annotations_path,
            input_annotations_path=input_annotations_path,
            input_items_path=input_items_path,
            upload_items=upload_items,
            add_labels_to_recipe=add_labels_to_recipe,
            concurrency=concurrency,
            return_error_filepath=return_error_filepath,
        )

    async def convert_dataset(self, labels_txt_filepath):
        """
        Converting a dataset from Yolo format to Dataloop.
        """
        # inputs
        self.label_txt_filepath = labels_txt_filepath
        # read labels and handle recipes
        with open(self.label_txt_filepath, 'r') as f:
            self.id_to_label_map = {i_label: label.strip() for i_label, label in enumerate(f.readlines())}
        if self.add_labels_to_recipe:
            self.dataset.add_labels(label_list=list(self.id_to_label_map.values()))

        # read annotations files and run on items
        files = list(Path(self.input_annotations_path).rglob('*.txt'))
        for txt_file in files:
            _ = await self.on_item(annotation_filepath=str(txt_file))

    @staticmethod
    def _get_video_data(input_filename):
        vid = cv2.VideoCapture(input_filename)
        width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
        dimensions = (width, height)
        fps = vid.get(cv2.CAP_PROP_FPS)
        vid.release()
        return dimensions, fps

    async def on_item(self, **context):
        """
        Converting an item from Yolo format to Dataloop.
        """
        annotation_filepath = context.get('annotation_filepath')
        with open(annotation_filepath, 'r') as f:
            lines = f.readlines()

        # find images with the same name (ignore image ext)
        relpath = os.path.relpath(annotation_filepath, self.input_annotations_path)
        filename, ext = os.path.splitext(relpath)
        input_filepaths = list(Path(os.path.join(self.input_items_path)).rglob(f'{filename}.*'))
        if len(input_filepaths) != 1:
            assert AssertionError

        # input filepath found
        input_filename = str(input_filepaths[0])
        remote_rel_path = os.path.relpath(input_filename, self.input_items_path)
        dirname = os.path.dirname(remote_rel_path)
        if self.upload_items:
            # TODO add overwrite as input arg
            item = self.dataset.items.upload(local_path=input_filename, remote_path=f'/{dirname}')
        else:
            try:
                item = self.dataset.items.get(filepath=f'/{remote_rel_path}')
            except dl.exceptions.NotFound:
                raise

        # get item dimensions ([width, height])
        if item.width is None or item.height is None:
            if "image" in item.mimetype:
                img = Image.open(input_filename)
                dimensions = img.size
                img.close()
            elif "video" in item.mimetype:
                dimensions, fps = self._get_video_data(input_filename=input_filename)
                item.metadata["fps"] = fps
            else:
                raise Exception(f'Unsupported item type: {item.mimetype}')
        else:
            dimensions = (item.width, item.height)

            # get item fps (for videos only)
            if "video" in item.mimetype and item.fps is None:
                _, fps = self._get_video_data(input_filename=input_filename)
                item.metadata["fps"] = fps

        # handle exif orientation
        if item.system.get('exif', {}).get('Orientation', 0) in [5, 6, 7, 8]:
            height, width = dimensions
        else:
            width, height = dimensions

        # parse the annotations and upload them to the item
        annotation_collection = item.annotations.builder()
        for annotation in lines:
            annotation_collection.add(**await self.on_annotation(
                item=item,
                annotation=annotation,
                width=width,
                height=height
            ))

        await item.annotations._async_upload_annotations(annotation_collection)

    def _get_annotation_definition(self, width, height, label_id, coordinates):
        # Box coordinates format: <x> <y> <width> <height>
        if len(coordinates) == 4:
            ann_def = self.on_box(
                width=width,
                height=height,
                label_id=label_id,
                coordinates=coordinates,
            )
        # Polygon coordinates format: <x1> <y1> <x2> <y2> <x3> <y3> ... (even number of coordinates)
        elif len(coordinates) > 4 and len(coordinates) % 2 == 0:
            ann_def = self.on_polygon(
                width=width,
                height=height,
                label_id=label_id,
                coordinates=coordinates,
            )
        else:
            ann_def = None

        return ann_def

    async def on_annotation(self, **context) -> dict:
        """
        Convert from YOLO format to DATALOOP format. Use this as conversion_func param for functions that ask for this param.

        **Prerequisites**: You must be an *owner* or *developer* to use this method.
        :param context: additional params
        :return: converted Annotation entity
        :rtype: dtlpy.entities.annotation.Annotation
        """
        item: dl.Item = context.get('item')
        width = context.get('width')
        height = context.get('height')

        if "image" in item.mimetype:
            annotation: str = context.get('annotation')
            line_data = annotation.strip().split(" ")

            # <label_id> <coordinates>
            label_id = int(line_data[0])
            coordinates = np.asarray(line_data[1:]).astype(float)

            ann_def = self._get_annotation_definition(
                width=width,
                height=height,
                label_id=label_id,
                coordinates=coordinates
            )
            if ann_def is not None:
                result = {
                    "annotation_definition": ann_def
                }
            else:
                raise Exception(f'Unsupported image annotation format: {annotation}')

        elif "video" in item.mimetype:
            annotation: str = context.get('annotation')
            line_data = annotation.strip().split(" ")

            # <frame_num> <object_id> <label_id> <coordinates>
            frame_num = int(line_data[0])
            object_id = line_data[1]
            label_id = int(line_data[2])
            coordinates = np.asarray(line_data[3:]).astype(float)

            ann_def = self._get_annotation_definition(
                width=width,
                height=height,
                label_id=label_id,
                coordinates=coordinates
            )
            if ann_def is not None:
                result = {
                    "annotation_definition": ann_def,
                    "object_id": object_id,
                    "frame_num": frame_num
                }
            else:
                raise Exception(f'Unsupported video annotation format for: {annotation}')

        else:
            raise Exception(f'Unsupported item type: {item.mimetype}')

        return result

    def on_box(self, **context):
        width = context.get('width')
        height = context.get('height')
        label_id = context.get('label_id')
        coordinates = context.get('coordinates')

        x, y, w, h = coordinates
        label = self.id_to_label_map.get(label_id, f'{label_id}_MISSING')
        x = x * width
        y = y * height
        w = w * width
        h = h * height
        left = x - (w / 2)
        right = x + (w / 2)
        top = y - (h / 2)
        bottom = y + (h / 2)
        ann_def = dl.Box(
            top=top,
            bottom=bottom,
            left=left,
            right=right,
            label=label
        )
        return ann_def

    def on_polygon(self, **context):
        width = context.get('width')
        height = context.get('height')
        label_id = context.get('label_id')
        coordinates = context.get('coordinates')

        coordinates = np.asarray(coordinates).reshape(-1, 2)
        coordinates[:, 0] = coordinates[:, 0] * width
        coordinates[:, 1] = coordinates[:, 1] * height
        label = self.id_to_label_map.get(label_id, f'{label_id}_MISSING')
        ann_def = dl.Polygon(
            geo=coordinates.tolist(),
            label=label
        )
        return ann_def

    # TODO: Check how to define customizable flag for this option
    def on_segmentation(self, **context):
        width = context.get('width')
        height = context.get('height')
        label_id = context.get('label_id')
        coordinates = context.get('coordinates')

        coordinates = np.asarray(coordinates).reshape(-1, 2)
        coordinates[:, 0] = coordinates[:, 0] * width
        coordinates[:, 1] = coordinates[:, 1] * height
        label = self.id_to_label_map.get(label_id, f'{label_id}_MISSING')
        ann_def = dl.Segmentation.from_polygon(
            geo=coordinates.tolist(),
            label=label,
            shape=(height, width)
        )
        return ann_def


class DataloopToYolo(BaseExportConverter):
    """
    Annotation Converter
    """

    async def on_dataset(self, **context) -> dict:
        """

        :param: local_path: directory to save annotations to
        :param context:
        :return:
        """
        from_path = self.dataset.download_annotations(local_path=self.input_annotations_path)
        json_path = Path(from_path).joinpath('json')
        files = list(json_path.rglob('*.json'))
        self.label_to_id_map = self.dataset.instance_map
        os.makedirs(self.output_annotations_path, exist_ok=True)
        sorted_labels = [k for k, v in sorted(self.label_to_id_map.items(), key=lambda item: item[1])]
        with open(os.path.join(self.output_annotations_path, 'labels.txt'), 'w') as f:
            f.write('\n'.join(sorted_labels))

        tic = time.time()
        for annotation_json_filepath in files:
            with open(annotation_json_filepath, 'r') as f:
                data = json.load(f)
            json_annotations = data.pop('annotations')
            item = dl.Item.from_json(_json=data,
                                     client_api=dl.client_api,
                                     dataset=self.dataset)
            annotations = dl.AnnotationCollection.from_json(_json=json_annotations, item=item)

            _ = await self.on_item_end(
                **await self.on_item(
                    **await self.on_item_start(item=item,
                                               dataset=self.dataset,
                                               annotations=annotations,
                                               to_path=os.path.join(self.output_annotations_path, 'annotations'))
                )
            )
        logger.info('Done converting {} items in {:.2f}[s]'.format(len(files), time.time() - tic))
        return context

    async def on_item(self, **context) -> dict:
        item = context.get('item')
        dataset = context.get('dataset')
        annotations = context.get('annotations')
        to_path = context.get('to_path')
        outputs = dict()
        item_yolo_strings = list()
        for i_annotation, annotation in enumerate(annotations.annotations):
            outs = {
                "dataset": dataset,
                "item": item,
                "width": item.width,
                "height": item.height,
                "annotation": annotation,
                "annotations": annotations
            }
            if annotation.type == dl.AnnotationType.BOX:
                outs = await self.on_annotation_end(
                    **await self.on_box(
                        **await self.on_annotation_start(**outs)))
            elif annotation.type == dl.AnnotationType.POLYGON:
                outs = await self.on_annotation_end(
                    **await self.on_polygon(
                        **await self.on_annotation_start(**outs)))
            elif annotation.type == dl.AnnotationType.SEGMENTATION:
                outs = await self.on_annotation_end(
                    **await self.on_segmentation(
                        **await self.on_annotation_start(**outs)))
            else:
                continue  # skip unsupported annotation types
            item_yolo_strings.append(outs.get('yolo_string'))
            outputs[annotation.id] = outs
        context['outputs'] = outputs
        name, ext = os.path.splitext(item.name)
        output_filename = os.path.join(to_path, item.dir[1:], name + '.txt')
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        with open(output_filename, 'w') as f:
            f.write('\n'.join(item_yolo_strings))
        return context

    async def on_box(self, **context) -> dict:
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

        if "image" in item.mimetype:
            x = (annotation.left + annotation.right) / 2.0
            y = (annotation.top + annotation.bottom) / 2.0
            w = annotation.right - annotation.left
            h = annotation.bottom - annotation.top
            x = x / width
            w = w / width
            y = y / height
            h = h / height

            label_id = self.label_to_id_map[annotation.label]

            # <label_id> <x> <y> <width> <height>
            yolo_string = f'{label_id} {x} {y} {w} {h}'
            context['yolo_string'] = yolo_string

        elif "video" in item.mimetype:
            box_yolo_string_list = list()

            frame_annotation: dl.entities.FrameAnnotation
            for frame_annotation in list(annotation.frames.values()):
                x = (frame_annotation.left + frame_annotation.right) / 2.0
                y = (frame_annotation.top + frame_annotation.bottom) / 2.0
                w = frame_annotation.right - frame_annotation.left
                h = frame_annotation.bottom - frame_annotation.top
                x = x / width
                w = w / width
                y = y / height
                h = h / height

                frame_num = frame_annotation.frame_num
                object_id = annotation.object_id
                label_id = self.label_to_id_map[frame_annotation.label]

                if object_id is None:
                    raise Exception(
                        f'Object ID is missing for the annotation: (ID: "{annotation.id}", Label: "{annotation.label}")'
                    )

                # <frame_num> <object_id> <label_id> <x> <y> <width> <height>
                box_yolo_string_list.append(f'{frame_num} {object_id} {label_id} {x} {y} {w} {h}')

            yolo_string = '\n'.join(box_yolo_string_list)
            context['yolo_string'] = yolo_string

        return context

    async def on_polygon(self, **context) -> dict:
        """
        Convert from DATALOOP format to YOLO format. Use this as conversion_func param for functions that ask for this param.
        **Prerequisites**: You must be an *owner* or *developer* to use this method.
        :param context:
                See below

        :Keyword Arguments:
            * *annotation* (``dl.Annotations``) -- the polygon annotations to convert
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

        if "image" in item.mimetype:
            coordinates_list = list()
            for coordinates in annotation.geo:
                coordinates_list.append(f'{coordinates[0] / width} {coordinates[1] / height}')
            coordinates_string = ' '.join(coordinates_list)

            label_id = self.label_to_id_map[annotation.label]

            # <label_id> <x1> <y1> <x2> <y2> <x3> <y3> ...
            yolo_string = f'{label_id} {coordinates_string}'
            context['yolo_string'] = yolo_string

        elif "video" in item.mimetype:
            polygon_yolo_string_list = list()

            frame_annotation: dl.entities.FrameAnnotation
            for frame_annotation in list(annotation.frames.values()):
                coordinates_list = list()
                for coordinates in annotation.geo:
                    coordinates_list.append(f'{coordinates[0] / width} {coordinates[1] / height}')
                coordinates_string = ' '.join(coordinates_list)

                frame_num = frame_annotation.frame_num
                object_id = annotation.object_id
                label_id = self.label_to_id_map[frame_annotation.label]

                # <frame_num> <object_id> <label_id> <x1> <y1> <x2> <y2> <x3> <y3> ...
                polygon_yolo_string_list.append(f'{frame_num} {object_id} {label_id} {coordinates_string}')

            yolo_string = '\n'.join(polygon_yolo_string_list)
            context['yolo_string'] = yolo_string

        return context

    async def on_segmentation(self, **context) -> dict:
        """
        Convert from DATALOOP format to YOLO format. Use this as conversion_func param for functions that ask for this param.
        **Prerequisites**: You must be an *owner* or *developer* to use this method.
        :param context:
                See below

        :Keyword Arguments:
            * *annotation* (``dl.Annotations``) -- the segmentation annotations to convert (exporting in polygon format)
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

        if "image" in item.mimetype:
            polygon_yolo_string_list = list()

            polygons = dl.Polygon.from_segmentation(
                mask=annotation.geo,
                label=annotation.label,
                epsilon=0,
                max_instances=None
            )
            if not isinstance(polygons, list):
                polygons = [polygons]

            for polygon in polygons:
                annotation._annotation_definition = polygon
                polygon_yolo_context = await self.on_polygon(
                    annotation=annotation,
                    item=item,
                    width=width,
                    height=height
                )

                # <label_id> <x1> <y1> <x2> <y2> <x3> <y3> ...
                polygon_yolo_string_list.append(polygon_yolo_context.get('yolo_string'))

            yolo_string = '\n'.join(polygon_yolo_string_list)
            context['yolo_string'] = yolo_string

        elif "video" in item.mimetype:
            logger.warning('Segmentation annotations are not supported for video items')

        return context
