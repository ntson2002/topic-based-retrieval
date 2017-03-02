import numpy as np
import lda
import lda.datasets
X = lda.datasets.load_reuters()

print X.shape
vocab = lda.datasets.load_reuters_vocab()
vocab1 = [vocab[i] for i in range(len(vocab)) if X[1][i] != 0]

print ' '.join(vocab1)

print type(vocab)
print len(vocab)
titles = lda.datasets.load_reuters_titles()
print X.shape
print X.sum()


model = lda.LDA(n_topics=20, n_iter=1500, random_state=1)
model.fit(X)  # model.fit_transform(X) is also available
topic_word = model.topic_word_
n_top_words = 8
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))

doc_topic = model.doc_topic_
for i in range(10):
    print("{} (top topic: {})".format(titles[i], doc_topic[i].argmax()))