from tokenizers import Tokenizer
from transformers import PreTrainedTokenizer
from transformers import BatchEncoding
import numpy as np
import torch
import os
import sentencepiece as spm

class SentencePieceTokenizer(PreTrainedTokenizer):
    def __init__(self, prefix='wd5m_with_pad', max_tokenize_length=75, pad_to_max = False):
        super()
        path = os.path.join('data/sentencepiece', prefix + '.model')
        self.sp = spm.SentencePieceProcessor(model_file=path)
        self._pad_token_id = self.sp['<pad>']
        self._eos_token_id = self.sp['</s>']
        self.max_tokenize_length = max_tokenize_length
        self.pad_to_max = pad_to_max
        if self.pad_to_max == True:
            print('Max length padding enabled (needed for TPU)')

    def make_attention_mask(self, encode_output, max_len):
        # padding is with zeros
        attention_mask = np.zeros((len(encode_output), max_len), dtype=np.long)
        for i, seq in enumerate(encode_output):
            length = min(max_len, len(seq))
            attention_mask[i][:length] = np.ones((length), dtype=np.long)
        return attention_mask

    # returns BatchEncoding
    def __call__(self, text, padding=True, truncation=True, max_length=128, return_tensors="pt"):
        out = self.sp.encode(text)
        # out is a list of list, need to pad and add eos token
        for x in out:
            x.append(self._eos_token_id)

        if self.pad_to_max:
            max_len = max_length
        else:
            max_len = min(max([len(x) for x in out]), self.max_tokenize_length)

        attention_mask = self.make_attention_mask(out, max_len)
        input_ids = np.ones((len(out), max_len), dtype=np.long) * self._pad_token_id
        for i, seq in enumerate(out):
            length = min(max_len, len(seq))
            input_ids[i][:length] = seq[:length]

        input_ids = torch.LongTensor(input_ids)
        attention_mask = torch.LongTensor(attention_mask)
        print(input_ids.shape, attention_mask.shape)
        data = {'input_ids': input_ids, 'attention_mask': attention_mask}
        return BatchEncoding(data)

    def batch_decode(self, input_ids, skip_special_tokens=True):
        input_ids = input_ids.tolist()
        #TODO: why need to do this?
        for i, x in enumerate(input_ids):
            if x[0] == 0:
                input_ids[i] = input_ids[i][1:]
        decoded = self.sp.decode(input_ids)
        out = []
        for s in decoded:
            s = s.replace('<pad>', '')
            out.append(s)
        return out

    @property
    def vocab_size(self) -> int:
        return len(self.sp)

    @property
    def pad_token_id(self) -> int:
        return self._pad_token_id

    @property
    def eos_token_id(self) -> int:
        return self._eos_token_id



def main():
    tokenizer = SentencePieceTokenizer()
    out = tokenizer(['hello world', 'fklsajd fkjsdfsa fjhs dj'])
    print(out.input_ids)
    # out.input_ids=out.input_ids.cuda()
    decoded = tokenizer.batch_decode(out.input_ids)
    print(decoded)
    print(tokenizer.vocab_size)
    print(tokenizer.pad_token_id)

if __name__ == "__main__":
    main()
