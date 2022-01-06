import unittest
from unittest.mock import mock_open, patch
import os
import sys
sys.path.append('../')
from preprocess import labels

class LabelsTest(unittest.TestCase):
    """
    Test Suite for labels.
    """
    @patch('preprocess.labels.json')
    def test_get_symbol_cluster_name(self, mocked_json):
        # test the json is called to load the file
        filename = '../preprocess/clustered_symbol_list.json'
        labels.get_symbol_cluster_name(filename)
        mocked_json.loads.assert_called_once()
    
    @patch('preprocess.labels.json')
    def test_load_symbol_cluster(self, mocked_json):
        # test the json is called to load the file
        filename = '../preprocess/clustered_symbol_list.json'
        word_to_id, id_to_symbol = labels.load_symbol_cluster(filename)
        mocked_json.loads.assert_called_once()
        # test the return values are functioning
        for k, v in word_to_id.items():
            self.assertEqual(id_to_symbol[v], k)
        for k, v in id_to_symbol.items():
            self.assertEqual(word_to_id[v], k)

    @patch('preprocess.labels.json')
    def test_load_symbols_annotation(self, mocked_json):
        # test the json is called to load the file
        filename = '../data/annotations/Symbols.json'
        labels.load_symbols_annotation(filename)
        mocked_json.load.assert_called_once()

    @patch('preprocess.labels.load_symbols_annotation')
    def test_build_label_encoder(self, mocked_load_method):
        # test the load method is called
        le_path = os.path.join('..', labels.le_path)
        filename = '../data/annotations/Symbols.json'
        labels.build_label_encoder(filename, le_path)
        mocked_load_method.assert_called_once_with(filename)
        # test the file is written
        m = mock_open()
        with patch('builtins.open', m) as mocked_open, patch('pickle.dumps') as mocked_dumps:
            labels.build_label_encoder(filename, le_path)
            m.assert_called_with(le_path, 'wb')
            file = mocked_open()
            file.write.assert_called_once()
            mocked_dumps.assert_called_once()


if __name__ == "__main__":
    # Create the test suite from the cases above.
    labels_testsuite = unittest.TestLoader().loadTestsFromTestCase(LabelsTest)
    # This will run the test suite.
    unittest.TextTestRunner(verbosity=2).run(labels_testsuite)