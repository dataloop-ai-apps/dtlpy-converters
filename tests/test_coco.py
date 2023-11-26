import os
import logging
import unittest
import asyncio
import dtlpy as dl
from dtlpyconverters.coco.coco_converters import CocoToDataloop, DataloopToCoco

logging.basicConfig(level=logging.DEBUG)
logging.getLogger('dtlpy').setLevel('WARNING')
logging.getLogger('filelock').setLevel('WARNING')
logging.getLogger('urllib3').setLevel('WARNING')


class TestCoco(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        dl.setenv('rc')
        project = dl.projects.get(project_name='test-converters-app')
        cls.dataset_name = 'to-delete-test-coco-conv'
        cls.dataset = project.datasets.create(dataset_name=cls.dataset_name)

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.dataset_name == cls.dataset.name:
            cls.dataset.delete(True, True)

    def test_coco_to_dtlpy(self):
        annotation_path = 'examples/coco/coco'
        coco_json_filename = 'annotations.json'
        images_path = 'examples/coco/images'

        conv = CocoToDataloop(input_annotations_path=annotation_path,
                              input_items_path=images_path,
                              dataset=self.dataset,
                              upload_items=True,
                              )
        loop = asyncio.get_event_loop()
        loop.set_debug(True)
        loop.run_until_complete(conv.convert_dataset(box_only=False,
                                                     coco_json_filename=coco_json_filename,
                                                     to_polygon=False))
        # self.assertEqual()

    def test_dtlpy_to_coco(self):
        project = dl.projects.get('test-converters-app')
        dataset = project.datasets.get('clone-test-coco-converters')
        output_annotations_path = 'tmp/coco'
        conv = DataloopToCoco(output_annotations_path=output_annotations_path,
                              download_items=False,
                              download_annotations=True,
                              dataset=dataset)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(conv.convert_dataset())
        # self.assertEqual()


if __name__ == '__main__':
    unittest.main()
