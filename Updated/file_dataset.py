import linecache
from torch.utils.data import Dataset
import numpy as np
    
class FileDataset(Dataset):
    def __init__(self, filename, vocab_file_name, char_file_name, max_uttr_num, max_uttr_len, max_persona_num, max_persona_len, max_response_num, max_response_len, max_word_len, dataset="personachat"):
        super(FileDataset, self).__init__()
        self._filename = filename
        self._vocab_file_name = vocab_file_name
        self._char_file_name = char_file_name
        self._max_uttr_num = max_uttr_num
        self._max_uttr_len = max_uttr_len
        self._max_persona_num = max_persona_num
        self._max_persona_len = max_persona_len
        self._max_response_num = max_response_num
        self._max_response_len = max_response_len
        self._max_word_len = max_word_len
        self._dataset = dataset
        with open(filename, "r") as f:
            self._total_data = len(f.readlines())
        self.vocab = self.load_vocab()
        self.char_vocab = self.load_char_vocab()
    
    def load_vocab(self):
        vocab = {"_pad_": 0, "_unk_": 1}
        with open(self._vocab_file_name, 'r', encoding='utf8') as f:
            for line in f:
                word = line.strip()
                if word not in vocab:
                    vocab[word] = len(vocab)
        return vocab

    def load_char_vocab(self):
        chars = {}
        with open(self._char_file_name, 'r', encoding='utf8') as f:
            for line in f:
                idx, char = line.strip().split('\t')
                chars[char] = int(idx)
        return chars
    
    def __getitem__(self, idx):
        line = linecache.getline(self._filename, idx + 1)
        line = line.strip().split("\t")
        context = line[0]
        utterances = (context + " ").split(' _eos_ ')[:-1]
        # utterances = [utterance + " _eos_" for utterance in utterances]
        utterances = utterances[-self._max_uttr_num:]
        us_vec = []
        us_mask = []
        us_chars = []
        # TODO: check pad last or pad first
        for utterance in utterances:
            u_tokens = utterance.split(' ')[:self._max_uttr_len]
            u_len = len(u_tokens)
            u_vec = [self.vocab.get(x, 1) for x in u_tokens]
            u_mask = [0] * len(u_vec)
            u_pad_len = self._max_uttr_len - u_len
            u_vec += [0] * u_pad_len
            u_mask += [1] * u_pad_len
            us_vec.append(u_vec)
            us_mask.append(u_mask)
            u_chars = []
            for token in u_tokens:
                token = token.lower()[:self._max_word_len]
                ch_vec = [self.char_vocab.get(ch, 0) for ch in token]
                ch_pad_len = self._max_word_len - len(ch_vec)
                ch_vec += [0] * ch_pad_len
                u_chars.append(ch_vec)
            u_chars += [[0] * self._max_word_len] * u_pad_len
            us_chars.append(u_chars)
        us_pad_num = self._max_uttr_num - len(utterances)
        us_pad_vec = [[0] * self._max_uttr_len] * us_pad_num
        us_pad_mask = [[1] * self._max_uttr_len] * us_pad_num
        us_pad_chars = [[[0] * self._max_word_len] * self._max_uttr_len] * us_pad_num
        us_vec = us_pad_vec + us_vec
        us_mask = us_pad_mask + us_mask
        us_chars = us_pad_chars + us_chars
        assert len(us_vec) == len(us_mask) == len(us_chars)

        responses = line[1].split("|")
        assert len(responses) == self._max_response_num
        rs_vec = []
        rs_mask = []
        rs_chars = []
        for response in responses:
            r_tokens = response.split(' ')[:self._max_uttr_len]
            r_len = len(r_tokens)
            r_vec = [self.vocab.get(x, 1) for x in r_tokens]
            r_mask = [0] * len(r_vec)
            r_pad_len = self._max_uttr_len - r_len
            r_vec += [0] * r_pad_len
            r_mask += [1] * r_pad_len
            rs_vec.append(r_vec)
            rs_mask.append(r_mask)
            r_chars = []
            for token in r_tokens:
                token = token.lower()[:self._max_word_len]
                ch_vec = [self.char_vocab.get(ch, 0) for ch in token]
                ch_pad_len = self._max_word_len - len(ch_vec)
                ch_vec += [0] * ch_pad_len
                r_chars.append(ch_vec)
            r_chars += [[0] * self._max_word_len] * r_pad_len
            rs_chars.append(r_chars)
        assert len(rs_vec) == len(rs_mask) == len(rs_chars)

        if self._dataset == "personachat":
            personas = line[4].split("|")[-self._max_persona_num:]
        elif self._dataset == "cmudog":
            personas = line[3].split("|")[-self._max_persona_num:]
        else:
            assert False
        ps_vec = []
        ps_mask = []
        ps_chars = []
        for persona in personas:
            p_tokens = persona.split(' ')[:self._max_persona_len]
            p_len = len(p_tokens)
            p_vec = [self.vocab.get(x, 1) for x in p_tokens]
            p_mask = [0] * len(p_vec)
            p_pad_len = self._max_persona_len - p_len
            p_vec += [0] * p_pad_len
            p_mask += [1] * p_pad_len
            ps_vec.append(p_vec)
            ps_mask.append(p_mask)
            p_chars = []
            for token in p_tokens:
                token = token.lower()[:self._max_word_len]
                ch_vec = [self.char_vocab.get(ch, 0) for ch in token]
                ch_pad_len = self._max_word_len - len(ch_vec)
                ch_vec += [0] * ch_pad_len
                p_chars.append(ch_vec)
            p_chars += [[0] * self._max_word_len] * p_pad_len
            ps_chars.append(p_chars)
        ps_pad_num = self._max_persona_num - len(personas)
        ps_pad_vec = [[0] * self._max_persona_len] * ps_pad_num
        ps_pad_mask = [[1] * self._max_persona_len] * ps_pad_num
        ps_pad_chars = [[[0] * self._max_word_len] * self._max_persona_len] * ps_pad_num
        ps_vec = ps_pad_vec + ps_vec
        ps_mask = ps_pad_mask + ps_mask
        ps_chars = ps_pad_chars + ps_chars        
        assert len(ps_vec) == len(ps_mask) == len(ps_chars)

        label = int(line[2])

        batch = {
            "ctx": np.asarray(us_vec),
            "doc": np.asarray(ps_vec),
            "rep": np.asarray(rs_vec),
            "ctx_char": np.asarray(us_chars),
            "doc_char": np.asarray(ps_chars),
            "rep_char": np.asarray(rs_chars),
            "ctx_mask": np.asarray(us_mask),
            "doc_mask": np.asarray(ps_mask),
            "rep_mask": np.asarray(rs_mask),
            "labels": label
        }

        return batch
    
    def __len__(self):
        return self._total_data

