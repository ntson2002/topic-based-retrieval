import query as api
from pprint import pprint


def printTopicVectors():
    print "================================================="
    print "List of topic vectors"
    topics = api.getTopics()
    for i in range(len(topics)):
        t = topics[i]

        words = [term[0] + "(" + str(term[1]) + ")" for term in t]
        print '\ttopic', i, ':', ' '.join(words)


def testQuery_TFIDF():
    print "================================================="
    print "Test query:"
    q = "At the request of the person provided in the main clause of paragraph 1 of Article 15 , or any assistant or supervisor of the assistant , the family court may make the ruling that the person under assistance must obtain the consent of his/her assistant if he/she intends to perform any particular juristic act ; provided , however , that the act for which such consent must be obtained pursuant to such ruling shall be limited to the acts provided in paragraph 1 of Article 13 ."
    model = "output/model_TFIDF.pkl"
    data = api.queryTFIDF(model, q, 10)
    pprint(data)


def testQuery_MDS():
    print "================================================="
    print "Test query on MDS space:"
    q = "At the request of the person provided in the main clause of paragraph 1 of Article 15 , or any assistant or supervisor of the assistant , the family court may make the ruling that the person under assistance must obtain the consent of his/her assistant if he/she intends to perform any particular juristic act ; provided , however , that the act for which such consent must be obtained pursuant to such ruling shall be limited to the acts provided in paragraph 1 of Article 13 ."
    model = "output/model_TFIDF_MDS.pkl"
    data = api.queryMDS(model, q, 10)
    pprint(data)


def testQuery_TopicBased():
    q = "At the request of the person provided in the main clause of paragraph 1 of Article 15 , or any assistant or supervisor of the assistant , the family court may make the ruling that the person under assistance must obtain the consent of his/her assistant if he/she intends to perform any particular juristic act ; provided , however , that the act for which such consent must be obtained pursuant to such ruling shall be limited to the acts provided in paragraph 1 of Article 13 ."
    top_k = 10
    model = "output/model_TFIDF.pkl"
    topicFile = "output/topic.pickle"
    data = api.queryTFIDF_topicBased(model, topicFile, q, 0, top_k)
    pprint(data)


if __name__ == '__main__':
    printTopicVectors()
    testQuery_TFIDF()
    testQuery_MDS()
    testQuery_TopicBased()

