import os
import pytest

from malcom import utils


class TestUtils():
    def test_json_reading(self):
        fn = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'data/test_json_reading')
        with open(fn) as input_file:
            ln = utils.Utils.readJsonObject(input_file)
            assert ln is not None
            ln = utils.Utils.readJsonObject(input_file)
            assert ln is not None
