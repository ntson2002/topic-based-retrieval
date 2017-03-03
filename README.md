### step 1. Prepare data
	A folder contains a list of legal articles 
	A json file contains all documents

### step 2. Document indexing
    1) Build TF-IDF vectors from corpus
    2) Build TF-IDF vectors from corpus then using MDS to reduce the space


    echo "=========================================="
    echo "Indexing TFIDF ..."
    INPUT=data/all_articles.json
    OUTPUT=output/model_TFIDF.pkl
    python document-indexing.py --index_type tfidf --file_type json --input $INPUT --output $OUTPUT

    echo "=========================================="
    echo "Indexing MDS ..."
    INPUT=data/all_articles.json
    OUTPUT=output/model_TFIDF_MDS.pkl
    python document-indexing.py --index_type mds --file_type json --input $INPUT --output $OUTPUT

### step 3. Build topic vectors from corpus
    1) Build topic model file from corpus

    echo "=========================================="
    echo "Creating topic vectors ..."
    INPUT=data/all_articles.json
    OUTPUT=output/topic.pickle
    python create-topic-model.py --file_type json --input $INPUT --output $OUTPUT

### step 4. Retrieval
    Support 3 type of query
    1) query on TF-IDF space
    2) query on MDS space (using MDS to reduce dimension)
    3) query on TF-IDF space with injection of topic vectors

### step 5. API
    1) start search api (default port = 8081)

    python search-api.py --port 8081

### step 6. Run API on web browser
    using "_" instead of spaces


    http://0.0.0.0:8081/api/search/A_demand_for_payment_shall_not_have_the_effect


