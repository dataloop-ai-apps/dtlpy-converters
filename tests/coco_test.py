import os
import unittest
import asyncio
import dtlpy as dl
from dataloop.converters.coco import CocoToDataloop, DataloopToCoco


class TestCoco(unittest.TestCase):

    def test_coco_to_dtlpy(self):
        project = dl.projects.get('test-converters-app')
        dataset = project.datasets.get('clone-test-coco-converters')
        annotation_filepath = '../examples/coco/coco/annotations.json'
        images_path = '../examples/coco/images'

        conv = CocoToDataloop()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(conv.convert_dataset(annotation_filepath=annotation_filepath,
                                                     images_path=images_path,
                                                     dataset=dataset,
                                                     upload_images=True,
                                                     box_only=False,
                                                     to_polygon=False))
        # self.assertEqual()

    def test_dtlpy_to_coco(self):
        project = dl.projects.get('test-converters-app')
        dataset = project.datasets.get('clone-test-coco-converters')
        to_path = '../tmp/coco'
        conv = DataloopToCoco()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(conv.convert_dataset(to_path=to_path,
                                                     download_images=False,
                                                     download_annotations=True,
                                                     dataset=dataset))
        # self.assertEqual()


if __name__ == '__main__':
    unittest.main()
