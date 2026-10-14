import logging

import torch
from tokenizers import Tokenizer


# Special tokens
SOT = "[START]"
EOT = "[STOP]"
UNK = "[UNK]"
SPACE = "[SPACE]"
SPECIAL_TOKENS = [SOT, EOT, UNK, SPACE, "[PAD]", "[SEP]", "[CLS]", "[MASK]"]

logger = logging.getLogger(__name__)

class MTLTokenizer:
    def __init__(self, vocab_file_path):
        self.tokenizer: Tokenizer = Tokenizer.from_file(vocab_file_path)
        self.check_vocabset_sot_eot()

    def check_vocabset_sot_eot(self):
        voc = self.tokenizer.get_vocab()
        assert SOT in voc
        assert EOT in voc

    def text_to_tokens(self, text: str, language_id: str = None):
        text_tokens = self.encode(text, language_id=language_id)
        text_tokens = torch.IntTensor(text_tokens).unsqueeze(0)
        return text_tokens

    def encode(self, txt: str, language_id: str = None, verbose=False):
        """
        clean_text > (prepend lang_id) > replace SPACE > encode text using Tokenizer
        
        The multilingual model expects language tokens like [en], [fr] to be prepended
        to condition the synthesis for the appropriate language.
        """
        # Prepend language token if provided
        if language_id:
            lang_token = f"[{language_id.lower()}]"
            txt = lang_token + txt
        
        txt = txt.replace(' ', SPACE)
        code = self.tokenizer.encode(txt)
        ids = code.ids
        return ids

    def decode(self, seq):
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu().numpy()

        txt: str = self.tokenizer.decode(seq,
        skip_special_tokens=False)
        txt = txt.replace(' ', '')
        txt = txt.replace(SPACE, ' ')
        txt = txt.replace(EOT, '')
        txt = txt.replace(UNK, '')
        return txt
