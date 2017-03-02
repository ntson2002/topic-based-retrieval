from function import *
from scipy.spatial import distance
import numpy as np
import scipy.spatial.distance as dis

# path_txt = '/Users/sonnguyen/Bitbucket/colliee2015-jaist/coliee2015/target/output/articles/*.txt'
# path_txt = 'data/example2/*.txt'
# files, vectorizer, X, datas = convert_to_tfidf_matrix(path_txt)

path_json = "/Users/sonnguyen/Dropbox/COLLIEE2015/data-2015/English_Training_Data/all_articles.json"
files, vectorizer, X, datas = convert_jsoncorpus_to_tfidf_matrix(path_json)
vocab = vectorizer.get_feature_names()

print type (files)
print type (vectorizer)
print type (X[0][0])
print type (datas)

D = distance.cdist(X, X, 'cosine')

print "Number of feature: ", len(vectorizer.get_feature_names())
print "Features:", vectorizer.get_feature_names()[2500:2510], "..."
print "Size of TF-IDF matrix: ", X.shape
print "Number of files: ", X.shape
print "Distance matrix shape: ", D.shape
print X[0]
print ' '.join([vocab[i] for i in range(len(X[1])) if X[1][i] != 0])
# docname = "/Users/sonnguyen/Bitbucket/colliee2015-jaist/coliee2015/target/output/articles/Article 427.txt"
# for doc in files:
#     print "------------------------------------------------"
#     print datas[doc]

import lda
model = lda.LDA(n_topics=10, n_iter=1500, random_state=1)
model.fit(X)  # model.fit_transform(X) is also available

topic_word = model.topic_word_
n_top_words = 30
for i, topic_dist in enumerate(topic_word):
    # print "sum...", sum(topic_dist)
    # print "topic_dist...", len(topic_dist), ".....:", topic_dist
    # print type (topic_dist)

    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    print zip(topic_words, 100*topic_dist[np.argsort(topic_dist)][:-(n_top_words+1):-1])
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))
    # print('Topic {}: {}'.format(i, topic_dist))

# print topic of documents
doc_topic = model.doc_topic_
for i in range(len(files)):
    # print doc_topic[i]
    dd = np.array([dis.cosine(topic_dist, X[i]) for j, topic_dist in enumerate(topic_word)])
    aa = dd.argmin()
    print "{} (top topic: {})".format(files[i], doc_topic[i].argmax()), aa, dd

print len(X[0])

for i, topic_dist in enumerate(topic_word):
    print len(topic_dist)
    print i, dis.cosine(topic_dist, X[0])







