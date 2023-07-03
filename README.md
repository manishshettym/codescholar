<img align="center" src="./codescholar.png"/>

How to run CodeScholar:
-----------------------


    
```bash
# start an elasticsearch server at port 9200
docker run --rm -p 9200:9200 -p 9300:9300 -e "xpack.security.enabled=false" -e "discovery.type=single-node" docker.elastic.co/elasticsearch/elasticsearch:8.7.0
```

```bash
# index the dataset using /search/elastic_search.py
cd codescholar/search
python elastic_search.py --dataset <dataset_name>
```

```bash
# run the codescholar query (say np.mean) using /search/search.py
python search.py --dataset <dataset_name> --seed np.mean
```

You can also use some arguments with the search query:
```bash
--min_idiom_size <int> # minimum size of idioms to be saved
--max_idiom_size <int> # maximum size of idioms to be saved
--max_init_beams <int> # maximum beams to initialize search
--stop_at_equilibrium  # stop search when diversity = reusability of idioms
```
*note: see more configurations in /search/search_config.py*

How to run CodeScholar Streamlit App:
---------------------------

```bash
# cd into the apps directory
cd codescholar/apps
```

```bash
# start a redis server to act as the message broker
docker run --rm -p 6379:6379 redis
```

```bash
# start a celery backend to handle tasks asynchronously
celery -A app_decl.celery worker --pool=solo --loglevel=info
```

```bash
# start a flask server to handle http API requests
# note: runs flask on port 3003
python app_main.py
```

You can now make API requests to the flask server. For example, to run search for size `10` idioms for `pd.merge`, you can:
```bash
curl -X POST -H "Content-Type: application/json" -d '{"api": "pd.merge", "size": 10}' http://localhost:3003/search
```

```bash
# start the streamlit app on port localhost:8501
streamlit run app_streamlit.py
```

<!-- Notes:
1. pd.map is not an API. It should be s.map for a series
2. pd.append is also not. It should be df.append

 -->