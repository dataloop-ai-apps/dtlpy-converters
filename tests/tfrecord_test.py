import dtlpy as dl
import unittest
import logging
import asyncio

from dataloop.converters.tfrecord import DataloopToTFRecord, TFRecordToDataloop

logging.basicConfig(level='INFO')


class TestTFRecord(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        dl.setenv('rc')
        project = dl.projects.get(project_name='test-converters-app')
        cls.dataset = project.datasets.get(dataset_name='to-delete-test-tfrecord-conv')

    @classmethod
    def tearDownClass(cls) -> None:
        cls.dataset.delete(True, True)

    def test_1_tfrecord_to_dtlpy(self):
        tfrecord_annotations_path = '../examples/tfrecord/tfrecord'
        images_path = '../examples/tfrecord/images'
        add_to_recipe = True

        conv = TFRecordToDataloop()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(conv.convert_dataset(annotations_path=tfrecord_annotations_path,
                                                     add_to_recipe=add_to_recipe,
                                                     images_path=images_path,
                                                     with_upload=True,
                                                     with_items=True,
                                                     dataset=self.dataset))
        # self.assertEqual()

    def test_2_dtlpy_to_tfrecord(self):
        images_path = '../tmp/voc/images'
        to_path = '../tmp/voc/voc'
        from_path = '../tmp/voc/dtlpy'

        conv = DataloopToTFRecord()
        loop = asyncio.get_event_loop()
        loop.run_until_complete(conv.convert_dataset(to_path=to_path,
                                                     from_path=from_path,
                                                     images_path=images_path,
                                                     download_binaries=True,
                                                     download_annotations=True,
                                                     dataset=self.dataset))
        # self.assertEqual()


if __name__ == '__main__':
    unittest.TestLoader.sortTestMethodsUsing = None
    unittest.main()
