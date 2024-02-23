#!/bin/bash

# Function to check if a tmux session exists
tmux_session_exists() {
    tmux has-session -t "$1" 2>/dev/null
}

# Function to start Elasticsearch in a tmux session
start_elasticsearch() {
    if tmux_session_exists "elasticsearch"; then
        echo "Elasticsearch tmux session already exists."
    else
        echo "Starting Elasticsearch in a tmux session..."
        tmux new-session -d -s "elasticsearch" docker run --name codescholar-elasticsearch --rm -p 9200:9200 -p 9300:9300 -e "xpack.security.enabled=false" -e "discovery.type=single-node" docker.elastic.co/elasticsearch/elasticsearch:8.7.0
    fi
}

# Function to start Redis in a tmux session
start_redis() {
    if tmux_session_exists "redis"; then
        echo "Redis tmux session already exists."
    else
        echo "Starting Redis in a tmux session..."
        tmux new-session -d -s "redis" docker run --name codescholar-redis --rm -p 6379:6379 redis
    fi
}

# Function to stop a service running in a tmux session
stop_service() {
    if tmux_session_exists "$1"; then
        echo "Stopping $1..."
        tmux send-keys -t "$1" C-c
        tmux kill-session -t "$1"
    else
        echo "No tmux session for $1 found."
    fi
}

# Function to run the indexing script
run_indexing() {
    dataset_name="$1"
    if [ -z "$dataset_name" ]; then
        echo "Please provide a dataset name."
        return 1
    fi
    echo "Running the indexing script for dataset: $dataset_name"
    tmux new-session -d -s "indexing" "cd search && python elastic_search.py --dataset $dataset_name"
}

# Function to display the status of the indexing session
show_indexing_status() {
    if tmux_session_exists "indexing"; then
        echo "Indexing session status:"
        tmux capture-pane -p -t "indexing"
    else
        echo "No indexing session found."
    fi
}

# Main logic based on the first argument
case "$1" in
    start)
        start_elasticsearch
        start_redis
        ;;
    stop)
        stop_service "elasticsearch"
        stop_service "redis"
        ;;
    killall)
        stop_service "elasticsearch"
        stop_service "redis"
        stop_service "indexing"
        ;;
    index)
        run_indexing "$2"
        ;;
    status)
        show_indexing_status
        ;;
    *)
        echo "Usage: $0 {start|stop|killall|index|status <dataset_name>}"
        exit 1
        ;;
esac