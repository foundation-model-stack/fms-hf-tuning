# First Party
from transformers import BertTokenizerFast

# Local
from .custom_tokenization import CustomTokenizer


class CustomTokenizerFast(BertTokenizerFast):
    slow_tokenizer_class = CustomTokenizer
    pass
