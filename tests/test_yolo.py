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
        annotations_path = 'examples/yolo/yolo/annotations'
        labels_txt_filepath = 'examples/yolo/yolo/labels.txt'
        images_path = 'examples/yolo/images'
        add_to_recipe = True

        conv = YoloToDataloop(input_annotations_path=annotations_path,
                              add_labels_to_recipe=add_to_recipe,
                              input_items_path=images_path,
                              upload_items=True,
                              dataset=self.dataset)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(conv.convert_dataset(labels_txt_filepath=labels_txt_filepath))
        # self.assertEqual()

    def test_2_dtlpy_to_yolo(self):
        images_path = 'tmp/yolo/images'
        to_path = 'tmp/yolo/yolo'
        from_path = 'tmp/yolo/dtlpy'

        conv = DataloopToYolo(output_annotations_path=to_path,
                              input_annotations_path=from_path,
                              output_items_path=images_path,
                              download_items=False,
                              dataset=self.dataset)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(conv.convert_dataset())
        # self.assertEqual()


if __name__ == '__main__':
    unittest.TestLoader.sortTestMethodsUsing = None
    unittest.main()
