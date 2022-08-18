import os
import unittest
import asyncio
import dtlpy as dl
from dataloop.converters.coco import CocoToDataloop, DataloopToCoco


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
        annotation_filepath = '../examples/coco/coco/annotations.json'
        images_path = '../examples/coco/images'

        conv = CocoToDataloop()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(conv.convert_dataset(annotation_filepath=annotation_filepath,
                                                     images_path=images_path,
                                                     dataset=self.dataset,
                                                     upload_images=True,
                                                     box_only=False,
                                                     to_polygon=False))
        # self.assertEqual()

    def test_dtlpy_to_coco(self):
        project = dl.projects.get('test-converters-app')
        dataset = project.datasets.get('clone-test-coco-converters')
        output_annotations_path = '../tmp/coco'
        conv = DataloopToCoco()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(conv.convert_dataset(output_annotations_path=output_annotations_path,
                                                     download_images=False,
                                                     download_annotations=True,
                                                     dataset=dataset))
        # self.assertEqual()


if __name__ == '__main__':
    unittest.main()
