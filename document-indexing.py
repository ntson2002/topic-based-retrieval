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


import optparse

if __name__ == '__main__':
    optparser = optparse.OptionParser()
    
    optparser.add_option(
        "-t", "--index_type", default="tfidf",
        help="{tfidf,mds} : indexing type"
    )

    optparser.add_option(
        "-f", "--file_type", default="json",
        help="{json,folder} : a json file contain all articles or a folder contains all txt files"
    )
    optparser.add_option(
        "-i", "--input", default="data/all_articles.json",
        help="path of json file or folder"
    )
    optparser.add_option(
        "-o", "--output", default="output/model_TFIDF.pkl",
        help="output"
    )

    # input = "data/all_articles.json"
    # output = 'output/model_TFIDF.pkl'
    # buildTFIDFModel(input, output)
    # buildMDSModel(input, output, 500)

    opts = optparser.parse_args()[0]
    print opts
    if opts.file_type == "json":
        if opts.index_type == "tfidf":
            buildTFIDFModel(opts.input, opts.output)
        elif opts.index_type == "mds":
            buildMDSModel(opts.input, opts.output, 500)
        else:
            print "Error index_type : ", opts.index_type
    else: #opts.file_type == "folder":
        print "coming soon"
