import os
import unittest
import sys
from unittest.mock import patch
import numpy as np
import torch
sys.path.append('../')
from preprocess import descriptors


class DescriptorsTest(unittest.TestCase):
    """
    Test Suite for descriptors.
    """
    @patch('preprocess.descriptors.api')
    def test_api(self, mock_api):
        # test api called once when model exists
        s = descriptors.SentimentPreProcessor(root="../data/annotations")
        if os.path.exists(s.embed_model):
            mock_api.load.assert_called_once_with(s.embed_model)

    def test_transform(self):
        s = descriptors.SentimentPreProcessor(root="../data/annotations")
        # case 1: all ids have the same frequency
        lst = [['1'], ['15'], ['30']]
        vec = s.transform(lst)
        np_array = np.array(s.model.get_vector(s.id_to_word[1]))
        torch.testing.assert_close(vec, torch.Tensor(np_array))
        # case 2: one id has the same frequency
        lst = [['1', '15'], ['15'], ['15', '30']]
        vec = s.transform(lst)
        np_array = np.array(s.model.get_vector(s.id_to_word[15]))
        torch.testing.assert_close(vec, torch.Tensor(np_array))
        # case 3: more than one id has the same frequency
        lst = [['1', '15', '30'], ['15', '30'], ['15', '30']]
        vec = s.transform(lst)
        np_array = np.array(s.model.get_vector(s.id_to_word[15]))
        torch.testing.assert_close(vec, torch.Tensor(np_array))
        


if __name__ == "__main__":
    # Create the test suite from the cases above.
    labels_testsuite = unittest.TestLoader().loadTestsFromTestCase(DescriptorsTest)
    # This will run the test suite.
    unittest.TextTestRunner(verbosity=2).run(labels_testsuite)