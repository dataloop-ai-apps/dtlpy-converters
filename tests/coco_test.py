import unittest

import dtlpy as dl
from converters.coco import CocoToDataloop, DataloopToCoco


class TestSum(unittest.TestCase):

    def test_coco_to_dtlpy(self):
        project = dl.projects.get('test-converters-app')
        dataset = project.datasets.get('test-coco-converters')
        annotation_filepath = '../converters/coco_converters/examples/coco/annotations.json'
        images_path = '../converters/coco_converters/examples/images'
        to_path = '../converters/coco_converters/examples/dataloop'

        conv = CocoToDataloop()
        conv.convert_dataset(annotation_filepath=annotation_filepath,
                             to_path=to_path,
                             images_path=images_path,
                             with_upload=True,
                             with_items=True,
                             dataset=dataset)
        # self.assertEqual()

    def test_dtlpy_to_coco(self):
        project = dl.projects.get('test-converters-app')
        dataset = project.datasets.get('test-coco-converters')
        images_path = '../tmp/images'
        to_path = '../tmp/coco'

        conv = DataloopToCoco()
        conv.convert_dataset(to_path=to_path,
                             images_path=images_path,
                             with_binaries=False,
                             dataset=dataset)
        # self.assertEqual()


if __name__ == '__main__':
    unittest.main()
