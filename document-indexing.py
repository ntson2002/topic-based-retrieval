from function import *
from scipy.spatial import distance
import numpy as np


def buildMDSModel(jsonCorpus, outPath, dimensions):
    files, vectorizer, matrix, datas = convert_jsoncorpus_to_tfidf_matrix(jsonCorpus)
    D = distance.cdist(matrix, matrix, 'cosine')
    M = convert_to_mds(D, dimensions)
    invD = np.linalg.inv(D)
    T = M.T.dot(invD)
    print 'Transpose operator T: ', T.shape
    save_object([files, vectorizer, matrix, D, T, M, invD, datas], outPath)
    print "model TF-IDF and MDS (dimensions=", dimensions, ") has been saved !!!"


def buildTFIDFModel(jsonCorpus, outPath):
    files, vectorizer, matrix, datas = convert_jsoncorpus_to_tfidf_matrix(jsonCorpus)
    # D = distance.cdist(matrix, matrix, 'cosine')

    print "Number of feature: ", len(vectorizer.get_feature_names())
    print "Size of TF-IDF matrix: ", matrix.shape
    print "Number of files: ", matrix.shape
    # print "Distance matrix: ", D.shape

    save_object([files, vectorizer, matrix, datas], outPath)
    print "model TF-IDF has been saved !!!"


if __name__ == '__main__':
    path_json = "data/all_articles.json"
    buildTFIDFModel(path_json, 'output/model_TFIDF.pkl')
    buildMDSModel(path_json, 'output/model_TFIDF_MDS.pkl', 500)
