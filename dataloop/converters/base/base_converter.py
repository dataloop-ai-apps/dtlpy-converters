from pathlib import Path
import dtlpy as dl
import logging
import time
import json

logger = logging.getLogger(name='dtlpy-converters')


class BaseConverter:
    """
    Annotation Converter
    """

    def __init__(self, concurrency=6, return_error_filepath=False):
        self.dataset = None
        self.concurrency = concurrency
        self.return_error_filepath = return_error_filepath

    async def convert_dataset(self, **kwargs):
        """
        :param dataset: dl.Dataset entity to convert
        :param kwargs:
        :return:
        """
        self.dataset = kwargs.get('dataset')
        return await self.on_dataset_end(
            ** await self.on_dataset(
                **await self.on_dataset_start(**kwargs)
            )
        )

    async def on_dataset_start(self, **kwargs):
        return kwargs

    async def on_dataset(self, **kwargs):
        """

        :param: local_path: directory to save annotations to
        :param dataset: dl.Dataset
        :param kwargs:
        :return:
        """
        dataset: dl.Dataset = kwargs.get('dataset')
        local_path = kwargs.get('local_path')
        dataset.download_annotations(local_path=local_path)
        json_path = Path(local_path).joinpath('json')
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
