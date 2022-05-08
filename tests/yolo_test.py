import unittest
import dtlpy as dl
from converters.yolo import YoloToDataloop, DataloopToYolo


class TestSum(unittest.TestCase):

    def test_yolo_to_dtlpy(self):
        dl.setenv('rc')
        project = dl.projects.get('test-converters-app')
        dataset = project.datasets.get('test-yolo-converters')
        annotations_path = '../converters/yolo/examples/yolo/annotations'
        label_txt_filepath = '../converters/yolo/examples/yolo/labels.txt'
        images_path = '../converters/yolo/examples/images'
        to_path = '../converters/yolo/examples/dataloop'
        add_to_recipe = True

        conv = YoloToDataloop()
        conv.convert_dataset(annotations_path=annotations_path,
                             label_txt_filepath=label_txt_filepath,
                             add_to_recipe=add_to_recipe,
                             to_path=to_path,
                             images_path=images_path,
                             with_upload=True,
                             with_items=True,
                             dataset=dataset)
        # self.assertEqual()

    def test_dtlpy_to_yolo(self):
        project = dl.projects.get('test-converters-app')
        dataset = project.datasets.get('test-yolo-converters')
        images_path = '../tmp/yolo/images'
        to_path = '../tmp/yolo/yolo'
        from_path = '../tmp/yolo/dtlpy'

        conv = DataloopToYolo()
        conv.convert_dataset(to_path=to_path,
                             from_path=from_path,
                             images_path=images_path,
                             with_binaries=False,
                             dataset=dataset)
        # self.assertEqual()


if __name__ == '__main__':
    unittest.main()
