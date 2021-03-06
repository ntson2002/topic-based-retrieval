### Step 1. Prepare data
    A folder contains a list of legal articles 
	A json file contains all documents

### Step 2. Document indexing
1) Build TF-IDF vectors from corpus

```sh
    echo "=========================================="
    echo "Indexing TFIDF ..."
    INPUT=data/all_articles.json
    OUTPUT=output/model_TFIDF.pkl
    python document-indexing.py --index_type tfidf --file_type json --input $INPUT --output $OUTPUT
```

2) Build TF-IDF vectors from corpus then using MDS to reduce the space
```sh
    echo "=========================================="
    echo "Indexing MDS ..."
    INPUT=data/all_articles.json
    OUTPUT=output/model_TFIDF_MDS.pkl
    python document-indexing.py --index_type mds --file_type json --input $INPUT --output $OUTPUT
```

### Step 3. Build topic vectors from corpus
1) Build topic model file from corpus

```sh
    echo "=========================================="
    echo "Creating topic vectors ..."
    INPUT=data/all_articles.json
    OUTPUT=output/topic.pickle
    python create-topic-model.py --file_type json --input $INPUT --output $OUTPUT
```
### Step 4. Retrieval
Support 3 types of query:

    1) query on TF-IDF space
    2) query on MDS space (using MDS to reduce dimension)
    3) query on TF-IDF space with injection of topic vectors

### Step 5. API
1) start search api (default port = 8081)

```sh
    $ python search-api.py --port 8081
```
### Step 6. Run API on web browser
Notes: using "_" instead of spaces
Query: A demand for payment shall not have the effect
URL:
    
    http://0.0.0.0:8081/api/search/A_demand_for_payment_shall_not_have_the_effect


### Notes 

1) Upload to server 

```sh
    git checkout master
    git add -A . && git commit -m "Upload"
    git push origin master 
```

1) Download from server

```sh
    git checkout master
    git pull origin master
```