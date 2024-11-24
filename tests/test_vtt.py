import unittest
import dtlpy as dl
import logging
import asyncio

from dtlpyconverters.vtt.vtt_converters import VttToDataloop, DataloopToVtt

logging.basicConfig(level='INFO')


class TestVtt(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        dl.setenv('rc')
        project = dl.projects.get(project_name='test-converters-app')
        cls.dataset = project.datasets.get(dataset_name='to-delete-test-vtt-conv')
        # cls.dataset = project.datasets.create(dataset_name='to-delete-test-vtt-conv')

    @classmethod
    def tearDownClass(cls) -> None:
        ...
        # cls.dataset.delete(True, True)

    def test_1_vtt_to_dtlpy(self):
        annotations_path = '../examples/vtt/vtt'
        audio_path = '../examples/vtt/audio'
        add_to_recipe = True

        conv = VttToDataloop(input_annotations_path=annotations_path,
                             add_labels_to_recipe=add_to_recipe,
                             input_items_path=audio_path,
                             upload_items=True,
                             dataset=self.dataset)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(conv.convert_dataset())
        # self.assertEqual()

    def test_2_dtlpy_to_vtt(self):
        audio_path = 'tmp/vtt/audio'
        to_path = 'tmp/vtt/vtt'
        from_path = 'tmp/vtt/dtlpy'

        conv = DataloopToVtt(output_annotations_path=to_path,
                             input_annotations_path=from_path,
                             output_items_path=audio_path,
                             download_items=False,
                             dataset=self.dataset)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(conv.convert_dataset())
        # self.assertEqual()


if __name__ == '__main__':
    unittest.TestLoader.sortTestMethodsUsing = None
    unittest.main()
