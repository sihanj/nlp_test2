import sys
import jieba
import os
import pandas as pd
import numpy as np
import _pickle as pickle
from sklearn.datasets.base import Bunch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# 保存至文件
def savefile(savepath, content):  
    with open(savepath, "w",encoding="utf-8") as fp:  
        fp.write(content)
        
def savefile_test(savepath, content):  
    with open(savepath, "a",encoding="utf-8") as fp:  
        fp.write(content+'\n')
        
# 读取文件       
def readfile(path):  
    with open(path, "r",encoding="utf-8") as fp:  
        content = fp.read() 
    return content 

def _readfile(path):  
    with open(path, "r",encoding="utf-8") as fp: 
        content = fp.read()  
    return content

# 读取bunch对象  
def _readbunchobj(path):  
    with open(path, "rb") as file_obj:  
        bunch = pickle.load(file_obj)  
    return bunch 

# 写入bunch对象  
def _writebunchobj(path, bunchobj):  
    with open(path, "wb") as file_obj:  
        pickle.dump(bunchobj, file_obj)

#P1_分词
#训练集
def corpus_segment(corpus_path, seg_path):
    catelist = os.listdir(corpus_path)
    for mydir in catelist:
        class_path = corpus_path + mydir + "/"
        seg_dir = seg_path + mydir + "/"
        if not os.path.exists(seg_dir):  # 是否存在目录，如果没有则创建
            os.makedirs(seg_dir)  
        file_list = os.listdir(class_path)
        for file_path in file_list:  # 遍历类别目录下的所有文件  
            fullname = class_path + file_path # 拼出文件名全路径
            content = readfile(fullname)
            content = content.replace("\r\n", "")  # 删除换行  
            content = content.replace(" ", "")#删除空行、多余的空格
            #jieba.load_userdict("dict.txt")
            content_seg = jieba.cut(content)  # 为文件内容分词
            savefile(seg_dir + file_path, " ".join(content_seg))  # 将处理后的文件保存到分词后语料目录  

if __name__=="__main__":  
    #对训练集进行分词  
    corpus_path = "./train_corpus/"  # 未分词分类语料库路径  
    seg_path = "./train_corpus_seg/"  # 分词后分类语料库路径  
    corpus_segment(corpus_path,seg_path)

#测试集
def corpus_segment_test(corpus_path, seg_path):
    file_list=os.listdir(corpus_path)
    for file_path in file_list:
        child=os.path.join(file_path)
        fullname=(corpus_path+'/'+child)
        fr = open(fullname,encoding='utf-8')
        fr_list = fr.read()
        dataList = fr_list.split('\n')
        contend_seg = []
        for oneline in dataList:
            #jieba.load_userdict("dict.txt")
            savefile_test(seg_path+file_path," ".join(jieba.cut(oneline)))
#中文语料分词结束！

if __name__=="__main__":     
    #对测试集进行分词  
    corpus_path = "test_corpus/"  # 未分词分类语料库路径  
    seg_path = "test_corpus_seg/"  # 分词后分类语料库路径  
    corpus_segment_test(corpus_path,seg_path)

#P2_构建文本对象bunch
def corpus2Bunch(wordbag_path,seg_path):  
    catelist = os.listdir(seg_path)# 获取seg_path下的所有子目录 
    #创建一个Bunch实例  
    bunch = Bunch(target_name=[], label=[], filenames=[], contents=[])  ################
    bunch.target_name.extend(catelist)
    for mydir in catelist:  
        class_path = seg_path + mydir + "/"  # 拼出分类子目录的路径  
        file_list = os.listdir(class_path)  # 获取class_path下的所有文件  
        for file_path in file_list:  # 遍历类别目录下文件  
            fullname = class_path + file_path  # 拼出文件名全路径  
            bunch.label.append(mydir)  
            bunch.filenames.append(fullname)  
            bunch.contents.append(_readfile(fullname))  # 读取文件内容  
    # 将bunch存储到wordbag_path路径中  
    with open(wordbag_path, "wb") as file_obj:  
        pickle.dump(bunch, file_obj)  
    print("构建文本对象结束！")  
    
def corpus2Bunch_test(wordbag_path,seg_path):  
    catelist = os.listdir(seg_path)
#创建Bunch 
    bunch = Bunch(target_name=[], label=[], contents=[])  #########################
    bunch.target_name.extend(catelist)
    for mydir in catelist:
        fullname=seg_path+"/"+mydir
        rows=open(fullname,"r",encoding="utf-8").read().splitlines()
        for file_path in rows:
            bunch.label.append(str(mydir).replace(".txt",""))
            bunch.contents.append(file_path)
    # 将bunch存储到wordbag_path 
    with open(wordbag_path, "wb") as file_obj:  
        pickle.dump(bunch, file_obj)  
    print("构建文本对象结束！")

if __name__ == "__main__":#这个语句前面的代码已经介绍过，今后不再注释  
    #对训练集进行Bunch化操作：  
    wordbag_path = "train_word_bag/train_set.dat"  # Bunch存储路径  
    seg_path = "train_corpus_seg/"  # 分词后分类语料库路径  
    corpus2Bunch(wordbag_path, seg_path)  
  
    # 对测试集进行Bunch化操作：  
    wordbag_path = "test_word_bag/test_set.dat"  # Bunch存储路径  
    seg_path = "test_corpus_seg/"  # 分词后分类语料库路径  
    corpus2Bunch_test(wordbag_path, seg_path)

 #P3_tf-idf词向量空间
 #这个函数用于创建TF-IDF词向量空间  
def vector_space(stopword_path,bunch_path,space_path,train_tfidf_path=None):  
  
    stpwrdlst = _readfile(stopword_path).splitlines()#读取停用词  
    bunch = _readbunchobj(bunch_path)#导入分词后的词向量bunch对象  
    #构建tf-idf词向量空间对象  
    tfidfspace = Bunch(target_name=bunch.target_name, label=bunch.label, filenames=bunch.filenames, tdm=[], vocabulary={})

    if train_tfidf_path is not None:  
        trainbunch = _readbunchobj(train_tfidf_path)  
        tfidfspace.vocabulary = trainbunch.vocabulary  
        vectorizer = TfidfVectorizer(stop_words=stpwrdlst, sublinear_tf=True, max_df=0.5,vocabulary=trainbunch.vocabulary)  
        tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)  
  
    else:  
        vectorizer = TfidfVectorizer(stop_words=stpwrdlst, sublinear_tf=True, max_df=0.5)  
        tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)  
        tfidfspace.vocabulary = vectorizer.vocabulary_ 
 
    _writebunchobj(space_path, tfidfspace)  
    print('tf-idf词向量空间实例创建成功！')  

def vector_space_test(stopword_path,bunch_path,space_path,train_tfidf_path=None):  
  
    stpwrdlst = _readfile(stopword_path).splitlines()  
    bunch = _readbunchobj(bunch_path)  
    tfidfspace = Bunch(target_name=bunch.target_name, label=bunch.label,file_contend=bunch.contents,tdm=[], vocabulary={})  
  
    if train_tfidf_path is not None:  
        trainbunch = _readbunchobj(train_tfidf_path)  
        tfidfspace.vocabulary = trainbunch.vocabulary  
        vectorizer = TfidfVectorizer(stop_words=stpwrdlst, sublinear_tf=True, max_df=0.5,vocabulary=trainbunch.vocabulary)  
        tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)  
  
    else:  
        vectorizer = TfidfVectorizer(stop_words=stpwrdlst, sublinear_tf=True, max_df=0.5)  
        tfidfspace.tdm = vectorizer.fit_transform(bunch.contents)  
        tfidfspace.vocabulary = vectorizer.vocabulary_  
  
    _writebunchobj(space_path, tfidfspace)  
    print ("if-idf词向量空间实例创建成功！")

if __name__ == '__main__':  
    stopword_path = "stopword.txt"#停用词表的路径  
    bunch_path = "train_word_bag/train_set.dat"  #导入训练集Bunch的路径  
    space_path = "train_word_bag/tfdifspace.dat"  # 词向量空间保存路径  
    vector_space(stopword_path,bunch_path,space_path)
    
    bunch_path = "test_word_bag/test_set.dat"  
    space_path = "test_word_bag/testspace.dat"  
    train_tfidf_path="train_word_bag/tfdifspace.dat"  
    vector_space_test(stopword_path,bunch_path,space_path,train_tfidf_path) 

    #P4_预测
    # 导入训练集  
trainpath = "train_word_bag/tfdifspace.dat"  
train_set = _readbunchobj(trainpath)  
  
# 导入测试集  
testpath = "test_word_bag/testspace.dat"  
test_set = _readbunchobj(testpath)  
  
# 训练分类器：输入词袋向量和分类标签，alpha:0.001 alpha越小，迭代次数越多，精度越高  
clf = MultinomialNB(alpha=0.001).fit(train_set.tdm, train_set.label)  
  
# 预测分类结果  
predicted = clf.predict(test_set.tdm)