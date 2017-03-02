import glob
import pickle
from sklearn import manifold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
import json


def getStopWords():
    list = stopwords.words('english')
    list.append("iv")
    list.append("v")
    list.append("iii")
    list.append("rrb")
    list.append("lrb")
    list.append("shall")
    list.append("may")
    list.append("article")
    list.append("paragraph")
    return list


def convert_jsoncorpus_to_tfidf_matrix(path):
    train_set = []
    datas = {}

    # load json file which contains all documents 
    with open(path) as data_file:
        jsonData = json.load(data_file)

    files = []  # list of file id or file name
    for i in range(len(jsonData)):
        files.append(jsonData[i]['article_id'])
        sentences = [item['text'] for item in jsonData[i]['sentences']]
        sentences.append(' ' + jsonData[i]['description'])
        content = ' '.join(sentences)
        train_set.append(content)
        datas[jsonData[i]['article_id']] = content

    # convert into TF-IDF vectors   
    stopWords = getStopWords()
    vectorizer = TfidfVectorizer(stop_words=stopWords, use_idf=True, sublinear_tf=False, norm='l2', smooth_idf=True)

    # vectorizer = CountVectorizer(stop_words = stopWords, min_df=1)
    trainVectorizerArray = vectorizer.fit_transform(train_set)

    return (files, vectorizer, trainVectorizerArray.toarray(), datas)


def convert_jsoncorpus_to_count_matrix(path):
    train_set = []
    datas = {}

    with open(path) as data_file:
        jsonData = json.load(data_file)

    files = []
    for i in range(len(jsonData)):
        files.append(jsonData[i]['article_id'])
        sentences = [item['text'] for item in jsonData[i]['sentences']]
        sentences.append(' ' + jsonData[i]['description'])
        content = ' '.join(sentences)
        train_set.append(content)
        datas[jsonData[i]['article_id']] = content

    stopWords = getStopWords()
    vectorizer = CountVectorizer(stop_words=stopWords, min_df=1)
    trainVectorizerArray = vectorizer.fit_transform(train_set)
    return (files, vectorizer, trainVectorizerArray.toarray(), datas)


def convert_to_tfidf_matrix(path):
    train_set = []
    datas = {}
    files = glob.glob(path)
    print 'Total files:', len(files)

    for file in files:
        f = open(file, 'r')
        data = f.readlines();
        content = ' '.join(data)  # join all line in a doc
        train_set.append(content)
        datas[file] = content
        f.close()

    stopWords = getStopWords()

    # vectorizer = TfidfVectorizer(stop_words = stopWords, use_idf=True, sublinear_tf=False , norm='l2', smooth_idf=True)
    vectorizer = CountVectorizer(stop_words=stopWords, min_df=1)
    trainVectorizerArray = vectorizer.fit_transform(train_set)
    return (files, vectorizer, trainVectorizerArray.toarray(), datas)


def query(test_query, files, vectorizer, matrix):
    testVectorizerArray = vectorizer.transform(test_query).toarray()
    scores = cosine_similarity(testVectorizerArray, matrix)
    sorted_by_second = sorted(zip(files, scores[0, :]), key=lambda tup: tup[1], reverse=True)
    return sorted_by_second


def transform_query_to_vector(vectorizer, test_query):
    return vectorizer.transform(test_query).toarray()


def convert_to_mds(matrix, dimensions=50):
    mds = manifold.MDS(n_components=dimensions, dissimilarity='precomputed')
    mdsMatrix = mds.fit_transform(matrix)
    return mdsMatrix


def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
