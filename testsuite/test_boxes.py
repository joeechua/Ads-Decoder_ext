import unittest
from unittest.mock import patch
import sys
sys.path.append('../')
from preprocess import boxes

class BoxesTest(unittest.TestCase):
    """
    Test Suite for boxes.
    """
    def test_load_symbols_annotation(self):
        with patch.object(boxes, 'load_symbols_annotation') as mock_load_method:
            # test the input argument is called once
            filename = 'symbols_json_file'
            mock_load_method(filename)
            mock_load_method.assert_called_once_with(filename)


if __name__ == "__main__":
    # Create the test suite from the cases above.
    boxes_testsuite = unittest.TestLoader().loadTestsFromTestCase(BoxesTest)
    # This will run the test suite.
    unittest.TextTestRunner(verbosity=2).run(boxes_testsuite)