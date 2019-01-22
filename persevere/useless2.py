
import pickle

y=pickle.load(open('./data/lables.pkl','rb'))

print(len(y[y==1]),len(y[y==0]))