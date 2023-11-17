import unittest
from unittest.mock import Mock, patch
import torch
from src.ablate_heads import BaseExperiment, TlensDataset, WrapHookedTransformer  # Import your classes

class TestBaseExperiment(unittest.TestCase):

    def setUp(self):
        # Setup mock dependencies
        self.mock_dataset = Mock(spec=TlensDataset)
        self.mock_model = Mock(spec=WrapHookedTransformer)
        self.batch_size = 32

        # Create an instance of BaseExperiment with mocked dependencies
        self.experiment = BaseExperiment(self.mock_dataset, self.mock_model, self.batch_size)

    def test_set_len_with_filter_outliers(self):
        # Test the set_len method with filter_outliers set to True
        test_length = 100
        self.experiment.filter_outliers = True
        self.experiment.set_len(test_length)

        # Assert that the dataset's length is set correctly and outliers are filtered
        self.mock_dataset.set_len.assert_called_with(test_length, self.mock_model)
        self.mock_dataset.slice_to_fit_batch.assert_not_called()  # Since outliers are filtered

    def test_set_len_without_filter_outliers(self):
        # Test the set_len method without filtering outliers
        test_length = 100
        self.experiment.filter_outliers = False
        self.experiment.set_len(test_length)

        # Assert that the dataset's length is set correctly and slicing is done
        self.mock_dataset.set_len.assert_called_with(test_length, self.mock_model)
        self.mock_dataset.slice_to_fit_batch.assert_called_with(self.batch_size)

# More test cases can be added here

if __name__ == '__main__':
    unittest.main()
