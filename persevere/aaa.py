from gensim import models
w2v_model = models.word2vec.Word2Vec.load('w2v.model')


if __name__ == '__main__':
    while True:
        s1=input('输入：')
        print(w2v_model.most_similar(positive=s1))