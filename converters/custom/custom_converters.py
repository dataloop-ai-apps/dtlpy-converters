import logging
import json
from pathlib import Path
import dtlpy as dl
from ..base import BaseConverter
import pandas as pd
import os

logger = logging.getLogger(name='dtlpy-converters')


class DataloopToCustomConverter(BaseConverter):
    """
    Annotation Converter
    """

    def __init__(self, concurrency=6, return_error_filepath=False):
        super(DataloopToCustomConverter, self).__init__(concurrency=concurrency,
                                                        return_error_filepath=return_error_filepath)
        self.dataset = None
        self.json_input = None
        self.concurrency = concurrency
        self.return_error_filepath = return_error_filepath
        self.outputs = dict()

    def create_csv_file(self, csv_filepath, dataset):
        # Changed the key in dict to be: item.dir/item.name+csv
        headers = list()
        for key in self.json_input.get('template'):
            headers.append(key)
        if 'item' in self.json_input.get('level'):
            for key, val in self.outputs.items():
                df = pd.DataFrame(data=list(val.values()),
                                  columns=headers)
                csv_file_name = os.path.join(csv_filepath, key)
                os.makedirs(csv_file_name, exist_ok=True)
                with open(csv_file_name, 'w') as f:
                    f.write(df.to_csv(index=False, line_terminator='\n'))

        elif 'dataset' in self.json_input.get('level'):
            lines_list = list()
            for item_id, lines in self.outputs.items():
                lines_list.append(pd.DataFrame(data=list(lines.values()),
                                               columns=headers))
            df = pd.concat(lines_list)
            csv_file_name = os.path.join(csv_filepath, '{}_{}.csv'.format(dataset.project.name,
                                                                          dataset.name))
            os.makedirs(csv_file_name, exist_ok=True)
            with open(csv_file_name, 'w') as f:
                f.write(df.to_csv(index=False, line_terminator='\n'))
        else:
            raise ValueError("Only dataset and item level supported for file outputs")

    async def convert_dataset(self, **kwargs):
        """
        :param dataset: dl.Dataset entity to convert
        :param kwargs:
        :return:
        """
        self.dataset = kwargs.get('dataset')
        json_input = kwargs.get('json_input')
        with open(json_input, 'r') as f:
            self.json_input = json.load(f)
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
        :param dataset: dl.Dataset
        :param kwargs:
        :return:
        """
        dataset: dl.Dataset = kwargs.get('dataset')
        local_path = kwargs.get('local_path')
        dataset.download_annotations(local_path=local_path)
        json_path = Path(local_path).joinpath('json')
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
                            dict_item_key=dict_item_key)))
        return kwargs

    async def on_dataset_end(self, **kwargs):
        self.create_csv_file(csv_filepath=kwargs.get('csv_file_path'), dataset=kwargs.get('dataset'))
        return kwargs

    async def on_item_start(self, **kwargs):
        return kwargs

    async def on_item(self, **kwargs):
        dataset = kwargs.get('dataset')
        item = kwargs.get('item')
        annotations = kwargs.get('annotations')
        dict_item_key = kwargs.get('dict_item_key')

        project = dataset.project
        if len(annotations) == 0:
            self.outputs[dict_item_key][item.id] = dict()
            if "csv" == self.json_input.get('output'):
                for header, value in self.json_input.get('template').items():
                    if 'frame' in value or 'annotation' in value:
                        self.outputs[dict_item_key][item.id][header] = None
                        continue
                    self.outputs[dict_item_key][item.id][header] = eval(value)
                return kwargs
            else:
                raise NotImplementedError('Support for Json outputs is not supported yet')
        for i_annotation, annotation in enumerate(annotations.annotations):
            if "video" in item.mimetype:
                frame_res = await self.on_annotation_end(
                    **await self.on_annotation(
                        **await self.on_annotation_start(
                            project=dataset.project,
                            item=item,
                            dataset=dataset,
                            annotation=annotation
                        )))
                self.outputs[dict_item_key].update(frame_res.get('output'))
            else:
                self.outputs[dict_item_key][annotation.id] = dict()
                if "csv" == self.json_input.get('output'):
                    for header, value in self.json_input.get('template').items():
                        if 'frame' in value:
                            self.outputs[dict_item_key][annotation.id][header] = None
                            continue
                        self.outputs[dict_item_key][annotation.id][header] = eval(value)
                else:
                    raise NotImplementedError('Support for Json outputs is not supported yet')
        return kwargs

    async def on_annotation(self, **kwargs):
        project = kwargs.get('project')
        dataset = kwargs.get('dataset')
        item = kwargs.get('item')
        annotation = kwargs.get('annotation')

        for frame_num in annotation.frames:
            kwargs['output'][(annotation.id, frame_num)] = dict()
            frame = annotation.frames[frame_num]
            kwargs['output'][(annotation.id, frame_num)] = dict()
            if "csv" == self.json_input.get('output'):
                for header, value in self.json_input.get('template').items():
                    kwargs['output'][(annotation.id, frame_num)][header] = eval(value)
            else:
                raise NotImplementedError('Support for Json outputs is not supported yet')

        return kwargs

    async def on_item_end(self, **kwargs):
        return kwargs

    async def on_annotation_start(self, **kwargs):
        if 'output' not in self.json_input.get('level'):
            kwargs['output'] = dict()
        return kwargs

    async def on_annotation_end(self, **kwargs):
        return kwargs
