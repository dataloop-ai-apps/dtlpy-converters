from jinja2 import Environment, PackageLoader
import xml.etree.ElementTree as Et
from pathlib import Path
import dtlpy as dl
import logging
import tqdm
import json
import os

from ..base import BaseConverter

logger = logging.getLogger(__name__)


class DataloopToVoc(BaseConverter):
    def __init__(self, concurrency=6, return_error_filepath=False):
        """
        Dataloop to COCO converter instance
        :param concurrency:
        :param return_error_filepath:


        """
        super(DataloopToVoc, self).__init__(concurrency=concurrency,
                                            return_error_filepath=return_error_filepath)

        # specific for voc
        labels = dict()
        # annotations template
        environment = Environment(loader=PackageLoader('converters', 'voc'),
                                  keep_trailing_newline=True)
        annotation_template = environment.get_template('voc_annotation_template.xml')
        self.annotation_params = {'labels': labels,
                                  'annotation_template': annotation_template}

    async def on_dataset(self, **kwargs):
        """
        Callback to tun the conversion on a dataset.
        Will be called after on_dataset_start and before on_dataset_end

        :param dataset:
        :param with_download:
        :param local_path:
        :param to_path:
        """
        download_binaries = kwargs.get('download_binaries')
        download_annotations = kwargs.get('download_annotations')
        dataset: dl.Dataset = kwargs.get('dataset')
        local_path = kwargs.get('local_path')
        to_path = kwargs.get('to_path')
        self.to_path_anns = os.path.join(to_path, 'annotations')
        self.to_path_masks = os.path.join(to_path, 'segmentation_class')

        if download_annotations:
            local_path = dataset.download_annotations(local_path=local_path)
            json_path = Path(local_path).joinpath('json')
        else:
            json_path = Path(local_path)
        if download_binaries:
            dataset.items.download(local_path=local_path)

        files = list(json_path.rglob('*.json'))
        self.pbar = tqdm.tqdm(total=len(files))
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
                                                   annotations=annotations)
                    )
                )

        return kwargs

    async def on_dataset_end(self, **kwargs):
        """
        """
        ...

    async def on_item(self, **kwargs):
        """

        :param item:
        :param annotations:
        """
        item: dl.Item = kwargs.get('item')
        annotations: dl.AnnotationCollection = kwargs.get('annotations')

        width = item.width
        height = item.height
        depth = item.metadata['system'].get('channels', 3)
        output_annotation = {
            'path': item.filename,
            'filename': os.path.basename(item.filename),
            'folder': os.path.basename(os.path.dirname(item.filename)),
            'width': width,
            'height': height,
            'depth': depth,
            'database': 'Unknown',
            'segmented': 0,
            'objects': list()
        }
        mask_annotations = dl.AnnotationCollection(item=item)
        for annotation in annotations:
            if annotation.type not in [dl.ANNOTATION_TYPE_BOX,
                                       dl.ANNOTATION_TYPE_SEGMENTATION,
                                       dl.ANNOTATION_TYPE_POLYGON]:
                continue
            if annotation.type in [dl.ANNOTATION_TYPE_SEGMENTATION, dl.ANNOTATION_TYPE_POLYGON]:
                output_annotation['segmented'] = 1
                mask_annotations.annotations.append(annotation)
                continue
            single_output_ann = await self.on_annotation_end(
                **await self.on_annotation(
                    **await self.on_annotation_start(annotation=annotation)
                )
            )
            output_annotation['objects'].append(single_output_ann)
        if output_annotation['segmented'] == 1:
            # download the masks
            out_filepath = os.path.join(self.to_path_masks, item.filename[1:])
            # remove ext from output filepath
            out_filepath, ext = os.path.splitext(out_filepath)
            # add xml extension
            out_filepath += '.png'
            os.makedirs(os.path.dirname(out_filepath), exist_ok=True)
            mask_annotations.download(filepath=out_filepath,
                                      annotation_format=dl.VIEW_ANNOTATION_OPTIONS_MASK)
        kwargs['output_annotation'] = output_annotation
        return kwargs

    async def on_item_end(self, **kwargs):
        """

        """
        item = kwargs.get('item')
        output_annotation = kwargs.get('output_annotation')

        # output filepath for xml
        out_filepath = os.path.join(self.to_path_anns, item.filename[1:])
        # remove ext from output filepath
        out_filepath, ext = os.path.splitext(out_filepath)
        # add xml extension
        out_filepath += '.xml'
        os.makedirs(os.path.dirname(out_filepath), exist_ok=True)
        with open(out_filepath, 'w') as file:
            content = self.annotation_params['annotation_template'].render(**output_annotation)
            file.write(content)
        self.pbar.update()
        return kwargs

    ##################
    # on annotations #
    ##################
    async def on_annotation(self, **kwargs):
        annotation: dl.Annotation = kwargs.get('annotation')
        single_output_ann = {'name': annotation.label,
                             'xmin': annotation.left,
                             'ymin': annotation.top,
                             'xmax': annotation.right,
                             'ymax': annotation.bottom,
                             'attributes': annotation.attributes,
                             }
        return single_output_ann


class VocToDataloop(BaseConverter):

    def __init__(self, concurrency=6, return_error_filepath=False):
        super(VocToDataloop, self).__init__(concurrency=concurrency,
                                            return_error_filepath=return_error_filepath)
        self.dataset = None
        self.concurrency = concurrency
        self.return_error_filepath = return_error_filepath

    async def convert_dataset(self, **kwargs):
        """

        """
        self.annotations_path = kwargs.get('annotations_path')
        self.images_path = kwargs.get('images_path')
        self.with_upload = kwargs.get('with_upload')
        self.with_items = kwargs.get('with_items')
        self.dataset = kwargs.get('dataset')
        xml_files = list(Path(self.annotations_path).rglob('*.xml'))
        self.pbar = tqdm.tqdm(total=len(xml_files))
        for annotation_xml_filepath in xml_files:
            filename = annotation_xml_filepath.relative_to(self.annotations_path)
            img_filepath = list(Path(self.images_path).glob(str(filename.with_suffix('.*'))))
            if len(img_filepath) > 1:
                raise ValueError(f'more than one image file with same name: {img_filepath}')
            elif len(img_filepath) == 0:
                img_filepath = None
            else:
                img_filepath = str(img_filepath[0])
            await self.on_item(img_filepath=img_filepath,
                               ann_filepath=annotation_xml_filepath)

    async def on_item(self, **kwargs):
        img_filepath = kwargs.get('img_filepath')
        ann_filepath = kwargs.get('ann_filepath')

        if self.with_upload:
            if img_filepath is None:
                logger.warning(f'could find local image for annotation file: {ann_filepath}')
                item = self.dataset.items.get(f'/{img_filepath}')
            else:
                item = self.dataset.items.upload(img_filepath)
        else:
            item = self.dataset.items.get(f'/{img_filepath}')
        with open(ann_filepath, "r") as f:
            voc_item = Et.parse(f)

        if voc_item.find('segmented') is not None and voc_item.find('segmented').text == '1':
            logger.warning('Only BB conversion is supported in VOC 2 DATALOOP. Segmentation will be ignored. Please contact support')

        voc_annotations = [e for e in voc_item.iter('object')]

        annotation_collection = item.annotations.builder()
        for voc_annotation in voc_annotations:
            out_args = await self.on_annotation_end(
                **await self.on_annotation(
                    **await self.on_annotation_start(**{'item': item,
                                                        'voc_annotation': voc_annotation})
                ))

            annotation_collection.annotations.append(out_args.get('dtlpy_ann'))
        item.annotations.upload(annotation_collection)

    async def on_annotation(self, **kwargs):
        """
        Convert from VOC format to DATALOOP format.

        **Prerequisites**: You must be an *owner* or *developer* to use this method.

        :param kwargs: additional params
        :return: converted Annotation entity
        :rtype: dtlpy.entities.annotation.Annotation
        """
        voc_annotation = kwargs.get('voc_annotation')
        item = kwargs.get('item')

        bndbox = voc_annotation.find('bndbox')

        if bndbox is None:
            raise Exception('No bndbox field found in annotation object')

        bottom = float(bndbox.find('ymax').text)
        top = float(bndbox.find('ymin').text)
        left = float(bndbox.find('xmin').text)
        right = float(bndbox.find('xmax').text)
        label = voc_annotation.find('name').text

        attributes = list(voc_annotation)
        attrs = list()
        for attribute in attributes:
            if attribute.tag not in ['bndbox', 'name'] and len(list(attribute)) == 0:
                if attribute.text not in ['0', '1']:
                    attrs.append(attribute.text)
                elif attribute.text == '1':
                    attrs.append(attribute.tag)

        ann_def = dl.Box(label=label,
                         top=top,
                         bottom=bottom,
                         left=left,
                         right=right,
                         attributes=attrs)
        kwargs['dtlpy_ann'] = dl.Annotation.new(annotation_definition=ann_def, item=item)
        return kwargs
