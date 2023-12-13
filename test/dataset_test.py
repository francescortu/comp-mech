import unittest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
sys.path.append("..")
sys.path.append("../src")
sys.path.append("../data")
from src.model import WrapHookedTransformer
from src.dataset import BaseDataset, TlensDataset, HFDataset

class TestDatasets(unittest.TestCase):
    def setUp(self):
        self.base_dataset = BaseDataset('../data/full_data_sampled_gpt2.json')
        self.model = WrapHookedTransformer()  # Initialize your model here
        self.tlens_dataset = TlensDataset(self.model, '../data/full_data_sampled_gpt2.json')
        self.hf_model = AutoModelForCausalLM.from_pretrained('gpt2')
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        self.hf_dataset = HFDataset(self.hf_model, self.tokenizer, '../data/full_data_sampled_gpt2.json')

    def test_len(self):
        with self.assertRaises(ValueError):
            len(self.base_dataset)
        self.base_dataset.set_len(10)
        self.assertEqual(len(self.base_dataset), 10)

    def test_getitem(self):
        with self.assertRaises(ValueError):
            self.base_dataset.__getitem__(0)
        self.base_dataset.set_len(10)
        item = self.base_dataset.__getitem__(0)
        self.assertIn('prompt', item)
        self.assertIn('input_ids', item)
        self.assertIn('target', item)
        self.assertIn('obj_pos', item)

    def test_tokenize_prompt_tlens(self):
        tokens = self.tlens_dataset._tokenize_prompt('Hello, world!', True)
        self.assertIsInstance(tokens, torch.Tensor)
        self.assertEqual(tokens.shape[0], 1)
        self.assertEqual(len(tokens.shape), 2)

    def test_tokenize_prompt_hf(self):
        tokens = self.hf_dataset._tokenize_prompt('Hello, world!', True)
        self.assertIsInstance(tokens, torch.Tensor)
        self.assertEqual(tokens.shape[0], 1)
        self.assertEqual(len(tokens.shape), 2)

if __name__ == '__main__':
    unittest.main()