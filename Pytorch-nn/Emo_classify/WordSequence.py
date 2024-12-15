class WordSequence:
    UNK_TAG = 'UNK'
    PAD_TAG = 'PAD'
    UNK = 0
    PAD = 1

    def __init__(self):
        self.word2index = {
            self.UNK_TAG : self.UNK,
            self.PAD_TAG : self.PAD
        }
        self.index2word = {
            self.UNK: self.UNK_TAG,
            self.PAD: self.PAD_TAG
        }
        self.count = {} 

    def fit(self, sentence):
        for word in sentence:
            self.count[word] = self.count.get(word,0) + 1

    def build_vocab(self, min_count = 0, max_feature = None):
        self.count = {key: value for key, value in self.count.items() if value >= min_count}
        if max_feature is not None:
            tmp = sorted(self.count.items(), key=lambda x: x[1], reverse=True)[:max_feature]
            self.count = {key:value for key, value in tmp}
        for word in self.count :
            self.word2index[word] = len(self.word2index)
            self.index2word[len(self.index2word)] = word
    
    def transform(self, sentence, max_len = None):
        indices = [self.word2index.get(word, self.UNK) for word in sentence]
        if max_len:
            if len(indices) < max_len:
                indices.extend([self.PAD] * (max_len - len(indices)))
            else:
                indices = indices[:max_len]
        return indices
    
    def inverse_transform(self, indices):
        sentence = [self.index2word.get(index)for index in indices]
        return sentence

if __name__ == "__main__":
    from WordSequence import WordSequence
    from Dataset import tokenization
    import os
    import pickle
    from tqdm import tqdm
    ws = WordSequence()
    path = r'Emo_classify/Imdb/train'
    temp_path = [os.path.join(path, 'pos'), os.path.join(path, 'neg')]
    for data_path in temp_path:
        file_name_list = [os.path.join(data_path, i) for i in os.listdir(data_path)]
        for file_path in tqdm(file_name_list):
            with open(file_path, 'r') as file:
                sentence = file.read()
                tokens = tokenization(sentence)
                ws.fit(tokens)
    ws.build_vocab(min_count = 10, max_feature = 10000)
    with open(r'Emo_classify/model/word_sequence.pkl', 'wb') as file:
        pickle.dump(ws, file)
    print(len(ws.word2index))
