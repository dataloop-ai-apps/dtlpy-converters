import os.path
import dtlpy as dl
import asyncio
from dtlpyconverters import coco_converters, yolo_converters, voc_converters
import nest_asyncio


class ConvertersUploader(dl.BaseServiceRunner):
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

    def coco_to_dataloop(self, dataset: dl.Dataset, input_annotations_path, input_items_path, coco_json_filename,
                         annotation_options=None, upload_items=True, to_polygon=True):
        """
        :param dataset: dataloop dataset
        :param input_annotations_path: path to annotations folder
        :param input_items_path: path to items folder
        :param coco_json_filename: coco json filename
        :param upload_items: upload items to dataloop
        :param annotation_options: list of annotation types to upload:
                                   [DEFAULT]: [dl.AnnotationType.BOX]
                                   [OPTIONS]: dl.AnnotationType.SEGMENTATION, dl.AnnotationType.BOX
        :param to_polygon: convert bbox to polygon
        """
        conv = coco_converters.CocoToDataloop(dataset=dataset,
                                              input_annotations_path=input_annotations_path,
                                              input_items_path=input_items_path,
                                              upload_items=upload_items
                                              )
        loop = self._get_event_loop()
        if annotation_options is None:
            annotation_options = [dl.AnnotationType.BOX]
        loop.run_until_complete(conv.convert_dataset(coco_json_filename=coco_json_filename,
                                                     annotation_options=annotation_options,
                                                     to_polygon=to_polygon))

    def yolo_to_dataloop(self, dataset: dl.Dataset, input_annotations_path, input_items_path, upload_items=True,
                         add_labels_to_recipe=True, labels_txt_filepath=None):
        """
        :param dataset: dataloop dataset
        :param input_annotations_path: path to annotations folder
        :param input_items_path: path to items folder
        :param upload_items: upload items to dataloop
        :param add_labels_to_recipe: Flag to add labels to recipe
        :param labels_txt_filepath: full path to labels txt file
        """
        conv = yolo_converters.YoloToDataloop(dataset=dataset,
                                              input_annotations_path=input_annotations_path,
                                              input_items_path=input_items_path,
                                              upload_items=upload_items,
                                              add_labels_to_recipe=add_labels_to_recipe)

        loop = self._get_event_loop()
        if not os.path.exists(labels_txt_filepath):
            raise Exception(f'file {labels_txt_filepath} file not found')
        loop.run_until_complete(conv.convert_dataset(labels_txt_filepath=labels_txt_filepath))

    def voc_to_dataloop(self, dataset: dl.Dataset, input_annotations_path, input_items_path, upload_items=True,
                        add_labels_to_recipe=True):
        """
        :param dataset: dataloop dataset
        :param input_annotations_path: path to annotations folder
        :param input_items_path: path to items folder
        :param upload_items: upload items to dataloop
        :param add_labels_to_recipe: Flag to add labels to recipe
        """
        conv = voc_converters.VocToDataloop(dataset=dataset,
                                            input_annotations_path=input_annotations_path,
                                            input_items_path=input_items_path,
                                            upload_items=upload_items,
                                            add_labels_to_recipe=add_labels_to_recipe)

        loop = self._get_event_loop()
        loop.run_until_complete(conv.convert_dataset())
