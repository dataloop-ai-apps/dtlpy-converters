from pathlib import Path
import pandas as pd
import dtlpy as dl
import logging
import json
import os

from ..base import BaseExportConverter

logger = logging.getLogger(name='dtlpy-converters')


class DataloopToCustomConverter(BaseExportConverter):
    """
    Annotation Converter
    """

    def __init__(self,
                 dataset: dl.Dataset,
                 output_annotations_path,
                 input_annotations_path=None,
                 output_items_path=None,
                 filters: dl.Filters = None,
                 download_annotations=True,
                 download_items=False,
                 concurrency=6,
                 return_error_filepath=False):
        """
        Convert Dataloop Dataset annotation to COCO format.

        :param dataset: dl.Dataset entity to convert
        :param output_annotations_path: where to save the converted annotations json
        :param input_annotations_path: where to save the downloaded dataloop annotations files. Default is output_annotations_path
        :param filters: dl.Filters object to filter the items from dataset
        :param download_items: download the images with the converted annotations
        :param download_annotations: download annotations from Dataloop or use local
        :return:
        """
        # global vars
        super(DataloopToCustomConverter, self).__init__(
            dataset=dataset,
            output_annotations_path=output_annotations_path,
            output_items_path=output_items_path,
            input_annotations_path=input_annotations_path,
            filters=filters,
            download_annotations=download_annotations,
            download_items=download_items,
            concurrency=concurrency,
            return_error_filepath=return_error_filepath,
        )
        self.outputs = dict()
        self.headers = list()

    def create_csv_file_per_item(self, csv_filepath, dataset):
        """
        Create a CSV file per item from a dict
        :param csv_filepath: local folder to save csv files
        :param dataset: dataset entity
        """
        for item_filepath, col_values in self.outputs.items():
            if col_values:
                df = pd.DataFrame(data=list(col_values.values()),
                                  columns=self.headers)
                csv_file_name = os.path.join(csv_filepath, item_filepath)
                os.makedirs(os.path.dirname(os.path.abspath(csv_file_name)), exist_ok=True)
                df.to_csv(path_or_buf=csv_file_name, index=False, line_terminator='\n')
                self.outputs[item_filepath] = None

    def create_csv_file_dataset(self, csv_filepath, dataset):
        """
        Create a CSV file per dataset from a dict
        :param csv_filepath: local folder to save csv files
        :param dataset: dataset entity
        """
        # create a csv file per dataset
        if 'dataset' in self.json_template.get('level'):
            lines_list = list()
            for item_filepath, col_values in self.outputs.items():
                for item_id, field_values in col_values.items():
                    lines_list.append(field_values)
            df = pd.DataFrame(data=lines_list, columns=self.headers)
            csv_file_name = os.path.join(csv_filepath, '{}_{}.csv'.format(dataset.project.name,
                                                                          dataset.name))
            os.makedirs(os.path.dirname(os.path.abspath(csv_file_name)), exist_ok=True)
            df.to_csv(path_or_buf=csv_file_name, index=False, line_terminator='\n')
        else:
            if not self.json_template.get('level') == 'item':
                raise ValueError("{} not supported,Only dataset and item level supported for file outputs".format(
                    self.json_template.get('level')))

    async def convert_dataset(self, json_template, **kwargs):
        """
        :param json_template: json file template for custom converter
        :return:
        """
        with open(json_template, 'r') as f:
            self.json_template = json.load(f)
        for key in self.json_template.get('template'):
            self.headers.append(key)
        return await self.on_dataset_end(
            **await self.on_dataset(
                **await self.on_dataset_start(**kwargs)
            )
        )

    async def on_dataset_start(self, **kwargs):
        return kwargs

    async def on_dataset(self, **kwargs):
        """
        :param dataset: dl.Dataset
        :param local_path: directory to save annotations to
        :param csv_file_path: local folder to save csv files
        :param kwargs:
        :return:
        """
        path = self.dataset.download_annotations(local_path=self.input_annotations_path)
        json_path = Path(path).joinpath('json')
        files = list(json_path.rglob('*.json'))

        for annotation_json_filepath in files:
            with open(annotation_json_filepath, 'r') as f:
                data = json.load(f)
            json_annotations = data.pop('annotations')
            item = dl.Item.from_json(_json=data,
                                     client_api=dl.client_api,
                                     dataset=self.dataset)

            name, ext = os.path.splitext(item.name)
            dict_item_key = os.path.join(item.dir[1:], name + '.csv')
            self.outputs[dict_item_key] = dict()

            annotations = dl.AnnotationCollection.from_json(_json=json_annotations, item=item)
            _ = await self.on_item_end(
                **await self.on_item(
                    **await self.on_item_start(
                        item=item,
                        dataset=self.dataset,
                        annotations=annotations,
                        dict_item_key=dict_item_key,
                        csv_file_path=self.output_annotations_path)))
        return kwargs

    async def on_dataset_end(self, **kwargs):
        self.create_csv_file_dataset(csv_filepath=kwargs.get('csv_file_path'), dataset=kwargs.get('dataset'))
        return kwargs

    async def on_item_start(self, **kwargs):
        return kwargs

    async def on_item(self, **kwargs):
        """
        :param dataset: dataset entity
        :param item: item entity
        :param annotations:
        :param dict_item_key: key for item in dict
        """
        item = kwargs.get('item')
        annotations = kwargs.get('annotations')
        dict_item_key = kwargs.get('dict_item_key')
        project = self.dataset.project

        # if item has no annotations, collect all the data that belongs to item/dataset/project entities only.
        if len(annotations) == 0:
            self.outputs[dict_item_key][item.id] = dict()
            if "csv" == self.json_template.get('output'):
                for header, value in self.json_template.get('template').items():
                    if 'frame' in value or 'annotation' in value:
                        self.outputs[dict_item_key][item.id][header] = None
                        continue
                    self.outputs[dict_item_key][item.id][header] = eval(value)
            elif 'json' == self.json_template.get('output'):
                raise NotImplementedError('Support for Json outputs is not supported yet')
            else:
                raise ValueError("{} filetype not supported".format(self.json_template.get('template')))
        else:
            for i_annotation, annotation in enumerate(annotations.annotations):
                # if the item is a video, go to on_annotation_start to collect annotation data per frame.
                if "video" in item.mimetype:
                    frame_res = await self.on_annotation_end(
                        **await self.on_annotation(
                            **await self.on_annotation_start(
                                project=self.dataset.project,
                                item=item,
                                dataset=self.dataset,
                                annotation=annotation
                            )))
                    self.outputs[dict_item_key].update(frame_res.get('output'))
                else:
                    # item is image, collect annotation data directly, and ignore frame keyword in template.
                    self.outputs[dict_item_key][annotation.id] = dict()
                    if "csv" == self.json_template.get('output'):
                        for header, value in self.json_template.get('template').items():
                            if 'frame' in value:
                                self.outputs[dict_item_key][annotation.id][header] = None
                                continue
                            self.outputs[dict_item_key][annotation.id][header] = eval(value)
                    elif 'json' == self.json_template.get('output'):
                        raise NotImplementedError('Support for Json outputs is not supported yet')
                    else:
                        raise ValueError("{} filetype not supported".format(self.json_template.get('template')))
        return kwargs

    async def on_annotation(self, **kwargs):
        """
        :param project: project entity
        :param dataset: dataset entity
        :param item: item entity
        :param annotation: annotation entity
        """
        project = kwargs.get('project')
        dataset = kwargs.get('dataset')
        item = kwargs.get('item')
        annotation = kwargs.get('annotation')

        for frame_num in annotation.frames:
            kwargs['output'][(annotation.id, frame_num)] = dict()
            frame = annotation.frames[frame_num]
            kwargs['output'][(annotation.id, frame_num)] = dict()
            if "csv" == self.json_template.get('output'):
                for header, value in self.json_template.get('template').items():
                    kwargs['output'][(annotation.id, frame_num)][header] = eval(value)
            elif 'json' == self.json_template.get('output'):
                raise NotImplementedError('Support for Json outputs is not supported yet')
            else:
                raise ValueError("{} filetype not supported".format(self.json_template.get('template')))

        return kwargs

    async def on_item_end(self, **kwargs):
        if self.json_template.get('level') == 'item':
            self.create_csv_file_per_item(csv_filepath=kwargs.get('csv_file_path'), dataset=kwargs.get('dataset'))
        return kwargs

    async def on_annotation_start(self, **kwargs):
        if 'output' not in self.json_template.get('level'):
            kwargs['output'] = dict()
        return kwargs

    async def on_annotation_end(self, **kwargs):
        return kwargs
