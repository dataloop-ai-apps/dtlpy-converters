import dtlpy as dl
from pathlib import Path
import logging
import tqdm
import json

from ..base import BaseImportConverter

logger = logging.getLogger('dtlpy-converter')


class LabelBoxToDataloop(BaseImportConverter):

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
        super(LabelBoxToDataloop, self).__init__(
            dataset=dataset,
            output_annotations_path=output_annotations_path,
            input_annotations_path=input_annotations_path,
            input_items_path=input_items_path,
            upload_items=upload_items,
            add_labels_to_recipe=add_labels_to_recipe,
            concurrency=concurrency,
            return_error_filepath=return_error_filepath,
        )

    async def convert_dataset(self):
        """

        """
        labelbox_files = list(Path(self.input_annotations_path).rglob('*.json'))
        pbar = tqdm.tqdm(total=len(labelbox_files))
        for filepath in labelbox_files:
            with open(filepath, 'r') as f:
                label_data = json.load(f)
            for data in label_data:
                await self.on_item(ann_data=data)
            pbar.update()

    async def on_item(self, **kwargs):
        ann_data = kwargs.get('ann_data')
        label_data = ann_data['Labeled Data']
        filename = ann_data['External ID']
        if self.upload_items:
            item = self.dataset.items.upload(local_path=label_data,
                                             remote_name=filename)
        else:
            item = self.dataset.items.get(f'/{filename}')

        annotation_collection: dl.AnnotationCollection = item.annotations.builder()
        if len(ann_data['Label']['objects']) != 0:
            for ann in ann_data['Label']['objects']:
                ann_def = await self.on_annotation(item=item,
                                                   labelbox_annotation=ann)
                if ann_def is not None:
                    annotation_collection.add(annotation_definition=ann_def)
        if len(ann_data['Label']['classifications']) != 0:
            for ann in ann_data['Label']['classifications']:
                label = ann['title']
                annotation_collection.add(annotation_definition=dl.Classification(label=label))

        item.annotations.upload(annotation_collection)

    async def on_annotation(self, **kwargs):
        """
        Convert from TFRecord format to DATALOOP format.

        **Prerequisites**: You must be an *owner* or *developer* to use this method.

        :param kwargs: additional params
        :return: converted Annotation entity
        :rtype: dtlpy.entities.annotation.Annotation
        """
        labelbox_annotation = kwargs.get('labelbox_annotation')
        label = labelbox_annotation['title']
        if 'polygon' in labelbox_annotation:
            geo = [[pt['x'], pt['y']] for pt in labelbox_annotation['polygon']]
            ann_def = dl.Polygon(label=label,
                                 geo=geo)

        elif 'point' in labelbox_annotation:
            ann_def = dl.Point(label=label,
                               x=labelbox_annotation['point']['x'],
                               y=labelbox_annotation['point']['y'], )

        elif 'line' in labelbox_annotation:
            geo = [[pt['x'], pt['y']] for pt in labelbox_annotation['line']]
            ann_def = dl.Polyline(label=label,
                                  geo=geo)

        elif 'bbox' in labelbox_annotation:
            bndbox = labelbox_annotation['bbox']

            if bndbox is None:
                raise Exception('No bndbox field found in annotation object')

            # upload box only
            left = bndbox['left']
            top = bndbox['top']
            right = left + bndbox['width']
            bottom = top + bndbox['height']

            ann_def = dl.Box(top=top,
                             left=left,
                             bottom=bottom,
                             right=right,
                             label=label)

        else:
            logger.warning('un-supported annotation type: {}'.format(labelbox_annotation))
            return

        return ann_def
