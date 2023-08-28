import dtlpy as dl
import unittest
import logging
import asyncio

from dtlpy_converters.tfrecord.tfrecord_converters import DataloopToTFRecord, TFRecordToDataloop

logging.basicConfig(level='INFO')


class TestTFRecord(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        dl.setenv('rc')
        project = dl.projects.get(project_name='test-converters-app')
        cls.dataset = project.datasets.create(dataset_name='to-delete-test-tfrecord-conv')

    @classmethod
    def tearDownClass(cls) -> None:
        cls.dataset.delete(True, True)

    def test_1_tfrecord_to_dtlpy(self):
        tfrecord_annotations_path = 'examples/tfrecord/tfrecord'
        images_path = 'examples/tfrecord/images'
        add_to_recipe = True

        conv = TFRecordToDataloop(input_annotations_path=tfrecord_annotations_path,
                                  add_labels_to_recipe=add_to_recipe,
                                  input_items_path=images_path,
                                  upload_items=True,
                                  dataset=self.dataset)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(conv.convert_dataset())
        # self.assertEqual()

    def test_2_dtlpy_to_tfrecord(self):
        images_path = 'tmp/voc/images'
        to_path = 'tmp/voc/voc'
        from_path = 'tmp/voc/dtlpy'

        conv = DataloopToTFRecord(output_annotations_path=to_path,
                                  input_annotations_path=from_path,
                                  output_items_path=images_path,
                                  download_items=True,
                                  download_annotations=True,
                                  dataset=self.dataset)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(conv.convert_dataset())
        # self.assertEqual()


if __name__ == '__main__':
    unittest.TestLoader.sortTestMethodsUsing = None
    unittest.main()
