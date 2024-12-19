from pathlib import Path
import dtlpy as dl
import logging
import time
import json

logger = logging.getLogger(name='dtlpy-converters')


class BaseExportConverter:
    """
    Annotation Converter
    """

    def __init__(self,
                 dataset: dl.Dataset,
                 output_annotations_path,
                 output_items_path=None,
                 input_annotations_path=None,
                 filters: dl.Filters = None,
                 download_annotations=True,
                 download_items=False,
                 concurrency=6,
                 return_error_filepath=False):
        if output_items_path is None:
            output_items_path = output_annotations_path
        if input_annotations_path is None:
            input_annotations_path = output_annotations_path

        self.dataset = dataset
        self.output_annotations_path = output_annotations_path
        self.output_items_path = output_items_path
        self.input_annotations_path = input_annotations_path
        self.filters = filters
        self.download_annotations = download_annotations
        self.download_items = download_items
        self.concurrency = concurrency
        self.return_error_filepath = return_error_filepath

    async def convert_dataset(self, **kwargs):
        """
        :param kwargs:
        :return:
        """
        return await self.on_dataset_end(
            **await self.on_dataset(
                **await self.on_dataset_start(**kwargs)
            )
        )

    async def on_dataset_start(self, **kwargs):
        return kwargs

    async def on_dataset(self, **kwargs):
        """

        :param: local_path: directory to save annotations to
        :param kwargs:
        :return:
        """
        path = self.dataset.download_annotations(local_path=self.input_annotations_path)
        json_path = Path(path).joinpath('json')
        files = list(json_path.rglob('*.json'))

        tic = time.time()
        for annotation_json_filepath in files:
            with open(annotation_json_filepath, 'r') as f:
                data = json.load(f)
                json_annotations = data.pop('annotations')
                item = dl.Item.from_json(_json=data,
                                         client_api=dl.client_api,
                                         dataset=self.dataset)
                annotations = dl.AnnotationCollection.from_json(_json=json_annotations, item=item)

                context = await self.on_item_end(
                    **await self.on_item(
                        **await self.on_item_start(item=item,
                                                   dataset=self.dataset,
                                                   annotations=annotations)
                    )
                )
        logger.info('Done converting {} items in {:.2f}[s]'.format(len(files), time.time() - tic))
        return kwargs

    async def on_dataset_end(self, **kwargs):
        return kwargs

    async def on_item_start(self, **kwargs):
        return kwargs

    async def on_item(self, **kwargs):
        item = kwargs.get('item')
        dataset = kwargs.get('dataset')
        annotations = kwargs.get('annotations')
        outputs = dict()
        for i_annotation, annotation in enumerate(annotations.annotations):
            outs = {"dataset": dataset,
                    "item": item,
                    "annotation": annotation,
                    "annotations": annotations}
            outs = await self.on_annotation_start(**outs)
            if annotation.type == dl.AnnotationType.BOX:
                outs = await self.on_box(**outs)
            elif annotation.type == dl.AnnotationType.POSE:
                outs = await self.on_pose(**outs)
            elif annotation.type == dl.AnnotationType.POLYGON:
                outs = await self.on_polygon(**outs)
            elif annotation.type == dl.AnnotationType.SEGMENTATION:
                outs = await self.on_polygon(**outs)
            outs = await self.on_annotation_end(**outs)
            outputs[annotation.id] = outs
        kwargs['outputs'] = outputs
        return kwargs

    async def on_item_end(self, **kwargs):
        return kwargs

    async def on_annotation_start(self, **kwargs):
        return kwargs

    async def on_annotation_end(self, **kwargs):
        return kwargs

    ##################
    # on annotations #
    ##################
    async def on_point(self, **kwargs):
        return kwargs

    async def on_box(self, **kwargs):
        return kwargs

    async def on_segmentation(self, **kwargs):
        return kwargs

    async def on_polygon(self, **kwargs):
        return kwargs

    async def on_class(self, **kwargs):
        return kwargs

    async def on_pose(self, **kwargs):
        return kwargs


class BaseImportConverter:
    """
    Annotation Converter
    """

    def __init__(self,
                 dataset: dl.Dataset,
                 input_annotations_path,
                 output_annotations_path=None,
                 input_items_path: str = None,
                 upload_items=False,
                 add_labels_to_recipe=True,
                 concurrency=6,
                 return_error_filepath=False):
        if output_annotations_path is None:
            output_annotations_path = input_annotations_path
        if input_items_path is None:
            input_items_path = input_annotations_path

        self.dataset = dataset
        self.input_annotations_path = input_annotations_path
        self.output_annotations_path = output_annotations_path
        self.input_items_path = input_items_path
        self.upload_items = upload_items
        self.add_labels_to_recipe = add_labels_to_recipe
        self.concurrency = concurrency
        self.return_error_filepath = return_error_filepath
