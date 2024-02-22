#!/bin/bash

# Function to check if a tmux session exists
tmux_session_exists() {
    tmux has-session -t "$1" 2>/dev/null
}

# Function to start the Celery worker
start_celery() {
    if tmux_session_exists "celery"; then
        echo "Celery tmux session already exists."
    else
        echo "Starting Celery worker in a tmux session..."
        tmux new-session -d -s "celery" celery -A app_decl.celery worker --pool=solo --loglevel=info
    fi
}

# Function to start the Flask server
start_flask() {
    if tmux_session_exists "flask"; then
        echo "Flask tmux session already exists."
    else
        echo "Starting Flask server in a tmux session..."
        tmux new-session -d -s "flask" python flask_app.py
    fi
}

# Function to start the Streamlit app
start_streamlit() {
    if tmux_session_exists "streamlit"; then
        echo "Streamlit tmux session already exists."
    else
        echo "Starting Streamlit app..."
        tmux new-session -d -s "streamlit" streamlit run streamlit_app.py
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

# Function to display the status of the streamlit session
show_streamlit_status() {
    if tmux_session_exists "streamlit"; then
        echo "Streamlit session status:"
        tmux capture-pane -p -t "streamlit"
    else
        echo "No streamlit session found."
    fi
}

# Main logic based on the first argument
case "$1" in
    start)
        start_celery
        start_flask
        start_streamlit
        ;;
    stop)
        stop_service "celery"
        stop_service "flask"
        stop_service "streamlit"
        ;;
    show)
        show_streamlit_status
        ;;
    *)
        echo "Usage: $0 {start|stop}"
        exit 1
        ;;
esac