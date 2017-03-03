#!/bin/bash
INPUT=data/all_articles.json

echo "=========================================="
echo "Indexing TFIDF ..."
OUTPUT=output/model_TFIDF.pkl
python document-indexing.py --index_type tfidf --file_type json --input $INPUT --output $OUTPUT

echo "=========================================="
echo "Indexing MDS ..."
OUTPUT=output/model_TFIDF_MDS.pkl
python document-indexing.py --index_type mds --file_type json --input $INPUT --output $OUTPUT


echo "=========================================="
echo "Creating topic vectors ..."
OUTPUT=output/topic.pickle
python create-topic-model.py --file_type json --input $INPUT --output $OUTPUT
