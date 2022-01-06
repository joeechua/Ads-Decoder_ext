import unittest
from unittest.mock import patch
import os
import sys
sys.path.append('../')
from preprocess import labels

class LabelsTest(unittest.TestCase):
    """
    Test Suite for labels.
    """
    def test_get_symbol_cluster_name(self):
        with patch.object(labels, 'get_symbol_cluster_name') as mock_get_method:
            # test the input argument is called once
            filename = 'cluster_json_file'
            mock_get_method(filename)
            mock_get_method.assert_called_once_with(filename)
            # test the return value has the correct length
            mock_get_method.return_value = ['cluster_one', 'cluster_two', 'cluster_three']
            cluster_names = mock_get_method(filename)
            self.assertEqual(len(cluster_names), 3)
    
    def test_load_symbol_cluster(self):
        with patch.object(labels, 'load_symbol_cluster') as mock_load_method:
            # test the input argument is called once
            filename = 'cluster_json_file'
            mock_load_method(filename)
            mock_load_method.assert_called_once_with(filename)
            # test the return values are dictionaries
            mock_load_method.return_value = ({'cluster_one': 1}, {1: 'cluster_one'})
            word_to_id, id_to_word = mock_load_method(filename)
            self.assertEqual(word_to_id['cluster_one'], 1)
            self.assertEqual(id_to_word[1], 'cluster_one')

    def test_load_symbols_annotation(self):
        with patch.object(labels, 'load_symbols_annotation') as mock_load_method:
            # test the input argument is called once
            filename = 'symbols_json_file'
            mock_load_method(filename)
            mock_load_method.assert_called_once_with(filename)

    def test_build_label_encoder(self):
        # test the label encoder file has created
        self.assertTrue(os.path.exists(os.path.join('..', labels.le_path)))


if __name__ == "__main__":
    # Create the test suite from the cases above.
    labels_testsuite = unittest.TestLoader().loadTestsFromTestCase(LabelsTest)
    # This will run the test suite.
    unittest.TextTestRunner(verbosity=2).run(labels_testsuite)