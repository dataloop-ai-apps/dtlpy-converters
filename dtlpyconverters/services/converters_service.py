import os.path
import datetime
import dtlpy as dl
import asyncio
import shutil
from dtlpyconverters import coco_converters, yolo_converters, voc_converters
import zipfile
import nest_asyncio


class DataloopConverters(dl.BaseServiceRunner):
    def __init__(self):
        nest_asyncio.apply()

    @staticmethod
    def _zip_folder(folder_path, output_path):
        try:
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(folder_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arc_name = os.path.relpath(file_path, folder_path)
                        zipf.write(file_path, arc_name)
        except Exception as e:
            raise e

    @staticmethod
    def _gen_converter_inputs(query):
        if query is None:
            filters = dl.Filters()
        else:
            filters = dl.Filters(custom_filter=query)
        timestamp = int(datetime.datetime.now().timestamp())
        output_annotations_path = os.path.join(os.getcwd(), '{}_output'.format(timestamp))
        input_annotations_path = os.path.join(os.getcwd(), '{}_input'.format(timestamp))
        return filters, timestamp, output_annotations_path, input_annotations_path

    @staticmethod
    def _get_event_loop():
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError as e:
            if "no current event loop" in str(e):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            else:
                raise e
        return loop

    def _convert_dataset(self, conv, conv_type, output_annotations_path, timestamp, input_annotations_path):
        zip_path = ''
        loop = self._get_event_loop()
        try:
            loop.run_until_complete(conv.convert_dataset())
            zip_path = os.path.join(os.getcwd(), '{}_{}.zip'.format(conv_type, timestamp))
            self._zip_folder(folder_path=output_annotations_path, output_path=zip_path)
            item = conv.dataset.items.upload(local_path=zip_path,
                                             remote_path='/.dataloop/{}'.format(conv_type))

            return item
        except Exception as e:
            raise e
        finally:
            if os.path.exists(output_annotations_path):
                shutil.rmtree(output_annotations_path)
            if os.path.exists(input_annotations_path):
                shutil.rmtree(input_annotations_path)
            if os.path.exists(zip_path):
                os.remove(zip_path)

    def dataloop_to_coco(self, dataset: dl.Dataset, query=None, download_items=False, download_annotations=True):
        """
        :param dataset: dataloop dataset
        :param query: dataloop dql
        :param download_items: bool download items
        :param download_annotations: bool download annotations
        :return: item id
        """
        filters, timestamp, output_annotations_path, input_annotations_path = self._gen_converter_inputs(query)
        conv = coco_converters.DataloopToCoco(input_annotations_path=input_annotations_path,
                                              output_annotations_path=output_annotations_path,
                                              download_items=download_items,
                                              download_annotations=download_annotations,
                                              filters=filters,
                                              dataset=dataset)
        output = self._convert_dataset(conv=conv,
                                       conv_type='coco',
                                       output_annotations_path=output_annotations_path,
                                       timestamp=timestamp,
                                       input_annotations_path=input_annotations_path)
        return output

    def dataloop_to_yolo(self, dataset: dl.Dataset, query=None, download_items=False, download_annotations=True):
        """
        :param dataset: dataloop dataset
        :param query: dataloop dql
        :param download_items: bool download items
        :param download_annotations: bool download annotations
        :return: item id
        """
        filters, timestamp, output_annotations_path, input_annotations_path = self._gen_converter_inputs(query)
        conv = yolo_converters.DataloopToYolo(input_annotations_path=input_annotations_path,
                                              output_annotations_path=output_annotations_path,
                                              download_items=download_items,
                                              download_annotations=download_annotations,
                                              filters=filters,
                                              dataset=dataset)
        output = self._convert_dataset(conv=conv,
                                       conv_type='yolo',
                                       output_annotations_path=output_annotations_path,
                                       timestamp=timestamp,
                                       input_annotations_path=input_annotations_path)
        return output

    def dataloop_to_voc(self, dataset: dl.Dataset, query=None, download_items=False, download_annotations=True):
        """
        :param dataset: dataloop dataset
        :param query: dataloop dql
        :param download_items: bool download items
        :param download_annotations: bool download annotations
        :return: item id
        """
        filters, timestamp, output_annotations_path, input_annotations_path = self._gen_converter_inputs(query)
        conv = voc_converters.DataloopToVoc(input_annotations_path=input_annotations_path,
                                            output_annotations_path=output_annotations_path,
                                            download_items=download_items,
                                            download_annotations=download_annotations,
                                            filters=filters,
                                            dataset=dataset)

        output = self._convert_dataset(conv=conv,
                                       conv_type='voc',
                                       output_annotations_path=output_annotations_path,
                                       timestamp=timestamp,
                                       input_annotations_path=input_annotations_path)
        return output
