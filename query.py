from scipy.spatial import distance
from function import *
import json
import sys
import pickle
import numpy as np
from pprint import pprint
import random


def openTopicVectors():
    with open('output/topic.pickle', 'rb') as handle:
        [model, vocab] = pickle.load(handle)

    print len(vocab)

    topic_word = model.topic_word_
    n_top_words = 30
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
        # print zip(topic_words, 100*topic_dist[np.argsort(topic_dist)][:-(n_top_words+1):-1])
        print('Topic {}: {}'.format(i, ' '.join(topic_words)))


def getTopics():
    with open('output/topic.pickle', 'rb') as handle:
        [model, topic_vectors, vocab] = pickle.load(handle)

    n_top_words = 10
    data = {}
    for i in range(len(topic_vectors)):
        topic_dist = topic_vectors[i]
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
        a = zip(topic_words, topic_dist[np.argsort(topic_dist)][:-(n_top_words + 1):-1])
        b = [[t[0], t[1]] for t in a]
        data[i] = b
    return data


def queryMDS(model, query, top_k):
    """

    :param model: model file
    :param query: query
    :param top_k: top k (int) relevant documents
    :return:
    """
    query = query.replace("_", " ")

    with open(model, 'rb') as input:
        [files, vectorizer, matrix, D, T, M, invD, datas] = pickle.load(input)
    test_query = [query]
    query_vec = transform_query_to_vector(vectorizer, test_query)
    d = distance.cdist(query_vec, matrix, 'cosine')
    m = T.dot(d.T)
    scores2 = distance.cdist(m.T, M, 'cosine')
    results2 = sorted(zip(files, scores2[0, :]), key=lambda tup: tup[1], reverse=False)

    list = []
    data = {}

    for t in results2[:top_k]:
        x = {}
        x["path"] = t[0]
        x["terms"] = ""
        x["score"] = t[1]
        x["text"] = datas[t[0]]
        list.append(x)

    data["result"] = list

    return data


def queryTFIDF_topicBased(model, topicFile, query, topic, top_k):
    """

    :param model: path of TF-IDF vectors
    :param topicFile: path of topic files that contains a list of topics
    :param query: input query
    :param topic: index of topic
    :param top_k: return top k most relevant documents
    :return: data + coordinate to visuallize
    """
    query = query.replace("_", " ")
    with open(model, 'rb') as input:
        [files, vectorizer, matrix, datas] = pickle.load(input)

    with open(topicFile, 'rb') as handle:
        [_, topic_vectors, vocab] = pickle.load(handle)

    test_query = [query]
    query_vec = transform_query_to_vector(vectorizer, test_query)  # query vec is a numpy matrix
    topic_vec = np.array(topic_vectors[int(topic)])  # topic vec
    new_query_vec = query_vec * (topic_vec * 20)

    scores2 = distance.cdist(new_query_vec, matrix, 'cosine')  # ma tran khoang cach

    top_inds = np.argsort(scores2[0, :])[:top_k]
    docs = [files[i] for i in top_inds]
    mm = np.array([matrix[i] for i in top_inds])
    mm = np.append(mm, new_query_vec, axis=0)

    results2 = sorted(zip(files, scores2[0, :]), key=lambda tup: tup[1], reverse=False)

    # check score
    # scores3 = distance.cdist(new_query_vec, mm, 'cosine')
    # print scores3

    list = []
    data = {}

    for t in results2[:top_k]:
        x = {}
        x["path"] = t[0]
        x["terms"] = ""
        x["score"] = t[1]
        x["text"] = datas[t[0]]
        list.append(x)

    data["result"] = list

    D = distance.cdist(mm, mm, 'cosine')
    r = lambda: random.randint(0, 255)

    mds_xy = convert_to_mds(D, 2)
    map_info = [(mds_xy[i][0], mds_xy[i][1], docs[i], '#%02X%02X%02X' % (255, i * 8, 0), 20) for i in range(top_k)]
    map_info.append((mds_xy[top_k][0], mds_xy[top_k][1], "query", "#00f", 25))

    map_data = {}
    map_data["animation"] = {"duration": 10000}
    # map_data["datasets"] = [{"label":"ABC", "backgroundColor":'#%02X%02X%02X' % (r(),r(),r()), "data":[{"x":t[0], "y":t[1], "r":10}]} for t in mds_xy]
    map_data["datasets"] = [{"label": t[2], "backgroundColor": t[3], "data": [{"x": t[0], "y": t[1], "r": t[4]}]} for t
                            in map_info]

    # data["mds"] = mds_xy
    data["map_data"] = map_data

    # pprint (map_data)
    return data


def queryTFIDF(model, query, top_k):
    query = query.replace("_", " ")
    with open(model, 'rb') as input:
        [files, vectorizer, matrix, datas] = pickle.load(input)
        # [files, vectorizer, matrix, D, T, M, invD, datas] = pickle.load(input)

    test_query = [query]
    query_vec = transform_query_to_vector(vectorizer, test_query)
    scores2 = distance.cdist(query_vec, matrix, 'cosine')

    top_inds = np.argsort(scores2[0, :])[:top_k]
    docs = [files[i] for i in top_inds]
    mm = np.array([matrix[i] for i in top_inds])
    mm = np.append(mm, query_vec, axis=0)

    results2 = sorted(zip(files, scores2[0, :]), key=lambda tup: tup[1], reverse=False)

    list = []
    data = {}

    for t in results2[:top_k]:
        x = {}
        x["path"] = t[0]
        x["terms"] = ""
        x["score"] = t[1]
        x["text"] = datas[t[0]]
        list.append(x)

    data["result"] = list

    D = distance.cdist(mm, mm, 'cosine')
    mds_xy = convert_to_mds(D, 2)
    map_info = [(mds_xy[i][0], mds_xy[i][1], docs[i], '#%02X%02X%02X' % (0, i * 8, 0), 20) for i in range(top_k)]
    map_info.append((mds_xy[top_k][0], mds_xy[top_k][1], "query", "#00f", 25))
    map_data = {
        "animation": {"duration": 10000},
        "datasets": [{"label": t[2], "backgroundColor": t[3], "data": [{"x": t[0], "y": t[1], "r": t[4]}]} for t in
                     map_info]
    }
    data["map_data"] = map_data
    return data


if __name__ == '__main__':
    # openTopicVectors()

    q = "At the request of the person provided in the main clause of paragraph 1 of Article 15 , or any assistant or supervisor of the assistant , the family court may make the ruling that the person under assistance must obtain the consent of his/her assistant if he/she intends to perform any particular juristic act ; provided , however , that the act for which such consent must be obtained pursuant to such ruling shall be limited to the acts provided in paragraph 1 of Article 13 ."
    # model = "model_TFIDF_MDS.pkl"
    # queryMDS(model, q, 10)
    model = "model_TFIDF.pkl"
    data = queryTFIDF(model, q, 10)
    from pprint import pprint

    pprint(data)


    # def queryOnMDSSpace(path, modelPath):
    #     with open(modelPath, 'rb') as input:
    #         [files, vectorizer, matrix, D, T, M, invD, datas] = pickle.load(input)
    #     text_file = open("question.txt", "r")
    #     lines = text_file.readlines()
    #     dict = {}
    #     contents = []
    #     for i in range(len(lines)):
    #         values = lines[i].split("\t")
    #         dict[values[0]] = values[1]
    #         test_query =[values[1]]
    #         query_vec = transform_query_to_vector(vectorizer, test_query)
    #         d = distance.cdist(query_vec, matrix, 'cosine')
    #         m = T.dot(d.T)
    #         scores2 = distance.cdist(m.T, M, 'cosine')
    #         results2 = sorted(zip(files, scores2[0,:]), key=lambda tup: tup[1], reverse=False)
    #         for t in results2[:30]:
    #             filename = t[0].replace('/Users/sonnguyen/Bitbucket/colliee2015-jaist/coliee2015/target/output/articles/', '').replace(' ', '_').replace('.txt','')
    #             line = '\t'.join([values[0], filename, str(t[1])]) #score
    #             #print line
    #             contents.append(line)
    #
    #     print "#records: ", len(contents)
    #     with open(path, 'w') as outfile:
    #         outfile.write('\n'.join(contents))
    #         outfile.close()
    #
    #
    # def queryOnTFIDFSpace(path, modelPath):
    #     dict = {}
    #     contents = []
    #     with open(modelPath, 'rb') as input:
    #         [files, vectorizer, matrix, D, T, M, invD, datas] = pickle.load(input)
    #     text_file = open("question.txt", "r")
    #     lines = text_file.readlines()
    #
    #     for i in range(len(lines)):
    #         values = lines[i].split("\t")
    #         dict[values[0]] = values[1]
    #         test_query =[values[1]]
    #         results = query (test_query, files, vectorizer, matrix)
    #         for t in results[:30]:
    #             filename = t[0].replace('/Users/sonnguyen/Bitbucket/colliee2015-jaist/coliee2015/target/output/articles/', '').replace(' ', '_').replace('.txt','')
    #             line = '\t'.join([values[0], filename, str(t[1])]) #score
    #             #print line
    #             contents.append(line)
    #
    #     print "#records: ", len(contents)
    #     with open(path, 'w') as outfile:
    #         outfile.write('\n'.join(contents))
    #         outfile.close()
    #
    #
    # def queryOnDifferentMDSFiles():
    #     dArray = [2, 5, 10, 15, 20, 50, 80, 100, 150, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    #     for i in range(len(dArray)):
    #         modelPath = "model_" + str(dArray[i]) + ".pkl"
    #         mdsResultPath = "mds_result_" + str(dArray[i]) + ".txt"
    #         queryOnMDSSpace(mdsResultPath, modelPath)
    #         print 'Result with MDS retrieval method has been saved to:', mdsResultPath
