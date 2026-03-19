import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # YOUR CODE HERE
        self.word_to_id[self.pad_token] = 0
        self.word_to_id[self.unk_token] = 1
        self.word_to_id[self.bos_token] = 2
        self.word_to_id[self.eos_token] = 3

        self.id_to_word[0] = self.pad_token
        self.id_to_word[1] = self.unk_token
        self.id_to_word[2] = self.bos_token
        self.id_to_word[3] = self.eos_token
        
        words = []

        for text in texts:
            tmp = text.split()
            words.extend(tmp)
            
        unique_words = sorted(set(words))
        self.vocab_size = len(unique_words) + 4

        base_id = 4

        for i,word in enumerate(unique_words):
            id = base_id + i
            self.word_to_id[word] = id
            self.id_to_word[id] = word
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        # YOUR CODE HERE
        words = text.split()
        tokens = []

        for word in words:
            try:
                token = self.word_to_id[word]
            except KeyError:
                token = self.word_to_id[self.unk_token]
            tokens.append(token)

        return tokens
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        # YOUR CODE HERE
        words = []

        for id in ids:
            word = self.id_to_word[id]
            words.append(word)

        return ' '.join(words)
