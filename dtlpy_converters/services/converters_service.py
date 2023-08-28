import os.path
import datetime
import dtlpy as dl
import asyncio
import shutil
from dtlpy_converters import coco_converters, yolo_converters, voc_converters
import zipfile


class DataloopConverters(dl.BaseServiceRunner):
    def __init__(self):
        pass

    @staticmethod
    def _zip_folder(folder_path, output_path):
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(folder_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    arc_name = os.path.relpath(file_path, folder_path)
                    zipf.write(file_path, arc_name)

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

    def _convert_dataset(self, conv, conv_type, output_annotations_path, timestamp, input_annotations_path):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(conv.convert_dataset())
        zip_path = os.path.join(os.getcwd(), '{}_{}.zip'.format(conv_type, timestamp))
        self._zip_folder(folder_path=output_annotations_path, output_path=zip_path)
        item = conv.dataset.items.upload(local_path=zip_path,
                                         remote_path='/.dataloop/{}'.format(conv_type))

        if os.path.exists(output_annotations_path):
            shutil.rmtree(output_annotations_path)
        if os.path.exists(input_annotations_path):
            shutil.rmtree(input_annotations_path)
        if os.path.exists(zip_path):
            os.remove(zip_path)
        return item.id

    def dataloop_to_coco(self, dataset: dl.Dataset, query=None):
        filters, timestamp, output_annotations_path, input_annotations_path = self._gen_converter_inputs(query)
        conv = coco_converters.DataloopToCoco(output_annotations_path=output_annotations_path,
                                              download_items=False,
                                              download_annotations=True,
                                              filters=filters,
                                              dataset=dataset)
        output = self._convert_dataset(conv=conv,
                                       conv_type='coco',
                                       output_annotations_path=output_annotations_path,
                                       timestamp=timestamp,
                                       input_annotations_path=input_annotations_path)
        return output

    def dataloop_to_yolo(self, dataset: dl.Dataset, query=None):
        filters, timestamp, output_annotations_path, input_annotations_path = self._gen_converter_inputs(query)
        conv = yolo_converters.DataloopToYolo(output_annotations_path=output_annotations_path,
                                              download_items=False,
                                              download_annotations=True,
                                              filters=filters,
                                              dataset=dataset)
        output = self._convert_dataset(conv=conv,
                                       conv_type='yolo',
                                       output_annotations_path=output_annotations_path,
                                       timestamp=timestamp,
                                       input_annotations_path=input_annotations_path)
        return output

    def dataloop_to_voc(self, dataset: dl.Dataset, query=None):
        filters, timestamp, output_annotations_path, input_annotations_path = self._gen_converter_inputs(query)
        conv = voc_converters.DataloopToVoc(output_annotations_path=output_annotations_path,
                                            download_items=False,
                                            download_annotations=True,
                                            filters=filters,
                                            dataset=dataset)
        output = self._convert_dataset(conv=conv,
                                       conv_type='voc',
                                       output_annotations_path=output_annotations_path,
                                       timestamp=timestamp,
                                       input_annotations_path=input_annotations_path)
        return output


# Run locally
if __name__ == '__main__':
    dataset = dl.datasets.get(dataset_id='6110d173f2428d3b2ae2dafc')
    service = dl.services.get(service_id='64eb570567f0fa83d89ffaea')
    service.execute(execution_input={'dataset': dataset.id},
                    function_name='dataloop_to_coco',
                    project_id=dataset.project.id)
#     dl.setenv('prod')
#     dataset = dl.datasets.get(dataset_id='')
#     runner = DataloopConverters()
#     item_id = runner.dataloop_to_coco(dataset=dataset)
#     yolo_items = runner.dataloop_to_yolo(dataset=dataset)
#     voc_items = runner.dataloop_to_voc(dataset=dataset)


# TODO : check coco metadata
