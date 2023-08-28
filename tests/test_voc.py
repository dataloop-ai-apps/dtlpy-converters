import dtlpy as dl
import unittest
import logging
import asyncio

from dtlpy_converters.voc.voc_converters import VocToDataloop, DataloopToVoc

logging.basicConfig(level='INFO')


class TestVoc(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        dl.setenv('rc')
        project = dl.projects.get(project_name='test-converters-app')
        cls.dataset = project.datasets.create(dataset_name='to-delete-test-voc-conv')

    @classmethod
    def tearDownClass(cls) -> None:
        cls.dataset.delete(True, True)
    #
    # def test_1_voc_to_dtlpy(self):
    #     annotations_path = 'examples/voc/voc/annotations'
    #     images_path = 'examples/voc/images'
    #     to_path = 'examples/voc/dataloop'
    #     add_to_recipe = True
    #
    #     conv = VocToDataloop(input_annotations_path=annotations_path,
    #                          add_labels_to_recipe=add_to_recipe,
    #                          output_annotations_path=to_path,
    #                          input_items_path=images_path,
    #                          upload_items=True,
    #                          dataset=self.dataset)
    #     loop = asyncio.get_event_loop()
    #     loop.run_until_complete(conv.convert_dataset())
    #     # self.assertEqual()

    def test_2_dtlpy_to_voc(self):
        images_path = 'tmp/voc/images'
        to_path = 'tmp/voc/voc'
        from_path = 'tmp/voc/dtlpy'

        conv = DataloopToVoc(output_annotations_path=to_path,
                             output_items_path=images_path,
                             input_annotations_path=from_path,
                             download_items=False,
                             download_annotations=True,
                             dataset=self.dataset)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(conv.convert_dataset())
        # self.assertEqual()


if __name__ == '__main__':
    unittest.TestLoader.sortTestMethodsUsing = None
    unittest.main()
