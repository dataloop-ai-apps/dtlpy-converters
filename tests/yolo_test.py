import unittest
import dtlpy as dl
import logging
import asyncio

from dataloop.converters.yolo import YoloToDataloop, DataloopToYolo

logging.basicConfig(level='INFO')


class TestYolo(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        dl.setenv('rc')
        project = dl.projects.get(project_name='test-converters-app')
        cls.dataset = project.datasets.create(dataset_name='to-delete-test-yolo-conv')

    @classmethod
    def tearDownClass(cls) -> None:
        cls.dataset.delete(True, True)

    def test_1_yolo_to_dtlpy(self):
        annotations_path = '../examples/yolo/yolo/annotations'
        label_txt_filepath = '../examples/yolo/yolo/labels.txt'
        images_path = '../examples/yolo/images'
        to_path = '../examples/yolo/dataloop'
        add_to_recipe = True

        conv = YoloToDataloop()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(conv.convert_dataset(annotations_path=annotations_path,
                                                     label_txt_filepath=label_txt_filepath,
                                                     add_to_recipe=add_to_recipe,
                                                     to_path=to_path,
                                                     images_path=images_path,
                                                     with_upload=True,
                                                     with_items=True,
                                                     dataset=self.dataset))
        # self.assertEqual()

    def test_2_dtlpy_to_yolo(self):
        images_path = '../tmp/yolo/images'
        to_path = '../tmp/yolo/yolo'
        from_path = '../tmp/yolo/dtlpy'

        conv = DataloopToYolo()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(conv.convert_dataset(to_path=to_path,
                                                     from_path=from_path,
                                                     images_path=images_path,
                                                     with_binaries=False,
                                                     dataset=self.dataset))
        # self.assertEqual()


if __name__ == '__main__':
    unittest.TestLoader.sortTestMethodsUsing = None
    unittest.main()
