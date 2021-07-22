from tokenizers import Tokenizer
from transformers import PreTrainedTokenizer
from transformers import BatchEncoding
import numpy as np
import torch
import os

class UnigramTokenizer(PreTrainedTokenizer):
    def __init__(self, prefix):
        super()
        path = os.path.join('data/wordpiece', prefix, 'main.json')
        self.tokenizer = Tokenizer.from_file(path)
        self._pad_token_id = self.tokenizer.encode('<pad>').ids[0]
        self._eos_token_id = self.tokenizer.encode('</s>').ids[0]
        self.tokenizer.enable_padding(pad_id=self._pad_token_id, pad_token="<pad>")
        # self.vocab_size = 2000

    # returns BatchEncoding
    def __call__(self, text, padding=True, truncation=True, max_length=128, return_tensors="pt"):
        text_processed = []
        for line in text:
            text_processed.append(line + ' </s>')
        out = self.tokenizer.encode_batch(text_processed)
        input_ids = []
        attention_mask = []
        for encoding in out:
            input_ids.append(np.array(encoding.ids))
            attention_mask.append(np.array(encoding.attention_mask))
        input_ids = torch.LongTensor(np.stack(input_ids))
        attention_mask = torch.LongTensor(np.stack(attention_mask))
        data = {'input_ids': input_ids, 'attention_mask': attention_mask}
        return BatchEncoding(data)

    def batch_decode(self, input_ids, skip_special_tokens=True):
        input_ids = input_ids.tolist()
        decoded = self.tokenizer.decode_batch(input_ids)
        out = []
        for s in decoded:
            s = s.replace(' ##', '')
            out.append(s)
        return out

    @property
    def vocab_size(self) -> int:
        return len(self.tokenizer.get_vocab())

    @property
    def pad_token_id(self) -> int:
        return self._pad_token_id

    @property
    def eos_token_id(self) -> int:
        return self._eos_token_id



def main():
    tokenizer = UnigramTokenizer('codexm2000')
    out = tokenizer(['hello world', 'fklsajd fkjsdfsa fjhs dj'])
    print(out.input_ids)
    # out.input_ids=out.input_ids.cuda()
    decoded = tokenizer.batch_decode(out.input_ids)
    print(decoded)
    print(tokenizer.vocab_size)
    print(tokenizer.pad_token_id)

if __name__ == "__main__":
    main()
