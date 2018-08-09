import re
import jieba
import networkx as nx
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

#分句
def cut_sentence(sentence):
    if not isinstance(sentence,str):
        sentence=sentence.decode('utf-8')
    delimiters=frozenset(u'。？！')
    buf=[]
    for ch in sentence:
        buf.append(ch)
        if delimiters.__contains__(ch):
            yield ''.join(buf)
            buf=[]
    if buf:
        yield ''.join(buf)

#停用词
def load_stopwords(path='stopwords.txt'):
    with open(path,encoding='utf-8') as f:
        stopwords=filter(lambda x:x,map(lambda x:x.strip(),f.readlines()))
    #stopwords.extend([' ','\t','\n'])
    return frozenset(stopwords)

#分词
def cut_words(sentence):
    stopwords=load_stopwords()
    return filter(lambda x: not stopwords.__contains__(x),jieba.cut(sentence))

#摘要
def get_abstract(content,size=3):
    docs=list(cut_sentence(content))
    tfidf_model=TfidfVectorizer(tokenizer=jieba.cut,stop_words=load_stopwords())
    tfidf_matrix=tfidf_model.fit_transform(docs)
    normalized_matrix=TfidfTransformer().fit_transform(tfidf_matrix)
    similarity=nx.from_scipy_sparse_matrix(normalized_matrix*normalized_matrix.T)
    scores=nx.pagerank(similarity)
    tops=sorted(scores.items(),key=lambda x:x[1],reverse=True)
    size=min(size,len(docs))
    indices=list(map(lambda x:x[0],tops))[:size] #list
    return map(lambda idx:docs[idx],indices)

a=input('请输入文档：')
a= re.sub(u'[　, ]',u'',a)
print('摘要为：')
abs=[]
for i in get_abstract(a):
    abs.append(i)
print(str(abs).replace("'",'').replace(",",'').replace(" ","").replace("[","").replace("]",""))
input('任意键退出程序')