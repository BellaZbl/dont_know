from persevere import model6
from tensorflow.contrib import learn


vocab_processor=learn.preprocessing.VocabularyProcessor.restore('vocab_processor.model')
vocab_num=len(vocab_processor.vocabulary_)
siamese_cnn = model6.siameseTextCNN(30, 30, None, vocab_num, 200, 60, [2, 3, 5])
aa=siamese_cnn.y_coe
print(aa)