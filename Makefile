
# Data Ingestion
run-get-videos:
	uv run service/youtube_extraction/src/youtube_extraction/youtube_parser.py

run-chunk-videos:
	uv run service/youtube_extraction/src/youtube_extraction/youtube_process.py

run-create-search:
	uv run service/youtube_extraction/src/youtube_extraction/youtube_search.py

run-data-prep: run-get-videos run-chunk-videos run-create-search
	

# Run Agent
run-terminal-app:
	uv run terminal_chatbot.py
	
run-streamlit-app:
	uv run streamlit run app.py

run-main:
	uv run main.py

# Run Evals
run-ground-truth:
	export PYTHONPATH="$PYTHONPATH:$(pwd)" && uv run service/evals/src/evals/create_ground_truth.py

run-ground-truth-evals:
	export PYTHONPATH="$PYTHONPATH:$(pwd)" && uv run service/evals/src/evals/run_eval_agent.py
