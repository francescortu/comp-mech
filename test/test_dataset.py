import unittest
from unittest.mock import Mock, patch
import torch
from src.dataset import TlensDataset, WrapHookedTransformer
from unittest.mock import mock_open



class TestTlensDataset(unittest.TestCase):
    @patch('builtins.open', new_callable=mock_open, read_data='{"key": "value"}')
    @patch('json.load')
    def setUp(self, mock_json_load, mock_file):
        # Setup mock dependencies
        self.mock_model = Mock(spec=WrapHookedTransformer)
        self.mock_model.to_tokens.return_value = torch.tensor([1, 2])
        self.mock_model.to_str_tokens.return_value = ['token1', 'token2']
        
        # Add a tokenizer attribute to the mock_model
        self.mock_model.tokenizer = Mock()
        self.mock_model.tokenizer.pad_token = '<pad>'

        mock_json_load.return_value = [{'prompt': 'prompt1', 'target_new': 'value1', "corrupted_prompts": "corruptedprompt"}, {'prompt': 'prompt2', 'target_new': 'value2', "corrupted_prompts": "corruptedprompt"}]

        # Create an instance of TlensDataset with mocked dependencies
        self.dataset = TlensDataset('path/to/data', self.mock_model, None)
    def test_set_len(self):
        # Test the set_len method
        test_length = 2
        self.dataset.set_len(test_length)

        # Assert that the dataset's length is set correctly
        self.assertEqual(self.dataset.length, test_length)

    def test_filter_from_idx_all(self):
        # Test the filter_from_idx_all method
        test_index = [0, 1]
        self.dataset.filter_from_idx_all(test_index)

        # Assert that the dataset's data is filtered correctly
        self.assertEqual(len(self.dataset.data), len(test_index))

    def test_filter_from_idx_exclude(self):
        # Test the filter_from_idx method with exclude set to True
        test_index = [0]
        self.dataset.filter_from_idx(test_index, exclude=True)

        # Assert that the dataset's data is filtered correctly
        self.assertNotIn(0, self.dataset.data)

    def test_filter_from_idx_include(self):
        # Test the filter_from_idx method with exclude set to False
        test_index = [0]
        self.dataset.filter_from_idx(test_index, exclude=False)

        # Assert that the dataset's data is filtered correctly
        self.assertIn(0, self.dataset.data)

    def test_slice(self):
        # Test the slice method
        start, end = 0, 2
        self.dataset.slice(end, start)

        # Assert that the dataset's data is sliced correctly
        self.assertEqual(len(self.dataset.corrupted_prompts), end - start)

    def test_get_lengths(self):
        # Test the get_lengths method
        lengths = self.dataset.get_lengths()

        # Assert that the lengths are returned correctly
        self.assertEqual(lengths, list(self.dataset.data_per_len.keys()))

    def test_slice_to_fit_batch(self):
        # Test the slice_to_fit_batch method
        batch_size = 2
        self.dataset.slice_to_fit_batch(batch_size)

        # Assert that the dataset's data is sliced correctly
        self.assertEqual(len(self.dataset.corrupted_prompts) % batch_size, 0)

if __name__ == '__main__':
    unittest.main()