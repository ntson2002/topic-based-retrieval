from function import convert_jsoncorpus_to_count_matrix
from scipy.spatial import distance
import numpy as np
import scipy.spatial.distance as dis
import lda
import pickle

# path_txt = '/Users/sonnguyen/Bitbucket/colliee2015-jaist/coliee2015/target/output/articles/*.txt'
# path_txt = 'data/example2/*.txt'
# files, vectorizer, X, datas = convert_to_tfidf_matrix(path_txt)

def create_topic_vectors(path_json, save_path):    
    files, vectorizer, X, datas = convert_jsoncorpus_to_count_matrix(path_json)
    vocab = vectorizer.get_feature_names()
    D = distance.cdist(X, X, 'cosine')

    print "Number of feature: ", len(vectorizer.get_feature_names())
    print "Features:", vectorizer.get_feature_names()[2500:2510], "..."
    print "Size of TF-IDF matrix: ", X.shape
    print "Number of files: ", X.shape
    print "Distance matrix shape: ", D.shape

    model = lda.LDA(n_topics=10, n_iter=2000, random_state=1)
    model.fit(X)  # model.fit_transform(X) is also available

    topic_word = model.topic_word_
    topic_vectors = [topic_dist for i, topic_dist in enumerate(topic_word)]

    with open(save_path, 'wb') as handle:
        pickle.dump([model, topic_vectors, vocab], handle)

    print "Topic vectors and vocab saved !!!"

    # topic_word = model.topic_word_
    # n_top_words = 30
    # for i, topic_dist in enumerate(topic_word):
    #     # print "sum...", sum(topic_dist)
    #     # print "topic_dist...", len(topic_dist), ".....:", topic_dist
    #     # print type (topic_dist)
    #
    #     topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    #     print zip(topic_words, 100*topic_dist[np.argsort(topic_dist)][:-(n_top_words+1):-1])
    #     print('Topic {}: {}'.format(i, ' '.join(topic_words)))
    #     # print('Topic {}: {}'.format(i, topic_dist))
    #
    # # print topic of documents
    # doc_topic = model.doc_topic_
    # for i in range(len(files)):
    #     # print doc_topic[i]
    #     dd = np.array([dis.cosine(topic_dist, X[i]) for j, topic_dist in enumerate(topic_word)])
    #     aa = dd.argmin()
    #     print "{} (top topic: {})".format(files[i], doc_topic[i].argmax()), aa, dd
    #
    # print len(X[0])
    #
    # for i, topic_dist in enumerate(topic_word):
    #     print len(topic_dist)
    #     print i, dis.cosine(topic_dist, X[0])


import optparse

if __name__ == '__main__':
    # path_json = "data/all_articles.json"
    # save_path = "output/topic.pickle"
    # create_topic_vectors(path_json, save_path)

    optparser = optparse.OptionParser()

    optparser.add_option(
        "-f", "--file_type", default="json",
        help="{json,folder} : a json file contain all articles or a folder contains all txt files"
    )
    optparser.add_option(
        "-i", "--input", default="data/all_articles.json",
        help="path of json file or folder"
    )
    optparser.add_option(
        "-o", "--output", default="output/topic.pickle",
        help="output"
    )


    opts = optparser.parse_args()[0]
    print opts
    if opts.file_type == "json":
        create_topic_vectors(opts.input, opts.output)
    else:  # opts.file_type == "folder":
        print "coming soon"