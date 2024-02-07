import dtlpy as dl
import unittest
import logging
import asyncio

from dtlpyconverters.labelbox.labelbox_converters import LabelBoxToDataloop

logging.basicConfig(level='INFO')


class TestLabelbox(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        dl.setenv('rc')
        project = dl.projects.get(project_name='test-converters-app')
        cls.dataset = project.datasets.create(dataset_name='to-delete-test-labelbox-conv')

    @classmethod
    def tearDownClass(cls) -> None:
        cls.dataset.delete(True, True)

    def test_1_labelbox_to_dtlpy(self):
        labelbox_annotations_path = 'examples/labelbox'
        add_to_recipe = True

        conv = LabelBoxToDataloop(input_annotations_path=labelbox_annotations_path,
                                  upload_items=True,
                                  dataset=self.dataset)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(conv.convert_dataset())
        # self.assertEqual()


if __name__ == '__main__':
    unittest.TestLoader.sortTestMethodsUsing = None
    unittest.main()
