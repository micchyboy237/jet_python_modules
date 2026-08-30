# Basic usage

python main.py -q "top 20 ongoing isekai anime with episodes and release dates"

# Custom safety limits

python main.py -q "..." --max-scrapes 50 --max-inner-links 3 --max-top-results 15

# Use cheaper model for evaluation

python main.py -q "..." --llm-model gpt-4o-mini --answer-model gpt-4o
