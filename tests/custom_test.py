import unittest
import dtlpy as dl
from dataloop.converters.custom import DataloopToCustomConverter
import asyncio


def test_dtlpy_to_custom():
    if dl.token_expired():
        dl.login()

    dataset = dl.datasets.get(dataset_id='61daebc07266c0aa07f94f1d')
    json_input_path = r"../examples/custom/json_input_templates/csv_example.json"
    local_annotation_path = r'../examples/custom/annotations'
    csv_file_path = r'../examples/custom/output'

    conv = DataloopToCustomConverter()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(conv.convert_dataset(dataset=dataset,
                                                 json_input=json_input_path,
                                                 local_path=local_annotation_path,
                                                 csv_file_path=csv_file_path))


# class TestSum(unittest.TestCase):


if __name__ == '__main__':
    test_dtlpy_to_custom()
