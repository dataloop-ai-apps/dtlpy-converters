import dtlpy as dl
import asyncio
from dtlpyconverters import coco_converters, yolo_converters, voc_converters
import nest_asyncio


class ConvertersDownloader(dl.BaseServiceRunner):
    def __init__(self):
        nest_asyncio.apply()

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

    def dataloop_to_coco(self, dataset: dl.Dataset, output_annotations_path, input_annotations_path=None,
                         download_annotations=True, filters=None):
        """
        :param dataset: dataloop dataset
        :param input_annotations_path: path to annotations folder
        :param output_annotations_path:
        :param download_annotations:
        :param filters:
        """
        conv = coco_converters.DataloopToCoco(input_annotations_path=input_annotations_path,
                                              output_annotations_path=output_annotations_path,
                                              download_annotations=download_annotations,
                                              filters=filters,
                                              dataset=dataset)
        loop = self._get_event_loop()
        loop.run_until_complete(conv.convert_dataset())

    def dataloop_to_yolo(self, dataset: dl.Dataset, output_annotations_path, input_annotations_path=None,
                         download_annotations=True, filters=None):
        """
        :param dataset: dataloop dataset
        :param input_annotations_path: path to annotations folder
        :param output_annotations_path:
        :param download_annotations:
        :param filters:
        """
        conv = yolo_converters.DataloopToYolo(input_annotations_path=input_annotations_path,
                                              output_annotations_path=output_annotations_path,
                                              download_annotations=download_annotations,
                                              filters=filters,
                                              dataset=dataset)
        loop = self._get_event_loop()
        loop.run_until_complete(conv.convert_dataset())

    def dataloop_to_voc(self, dataset: dl.Dataset, output_annotations_path, input_annotations_path=None,
                        download_annotations=True, filters=None):
        """
        :param dataset: dataloop dataset
        :param input_annotations_path: path to annotations folder
        :param output_annotations_path:
        :param download_annotations:
        :param filters:
        """
        voc_converter = voc_converters.DataloopToVoc(input_annotations_path=input_annotations_path,
                                                     output_annotations_path=output_annotations_path,
                                                     download_annotations=download_annotations,
                                                     filters=filters,
                                                     dataset=dataset)
        loop = self._get_event_loop()
        loop.run_until_complete(voc_converter.convert_dataset())
