from jet.adapters.bertopic import BERTopic
import pandas as pd
from typing import Tuple, Any
from datetime import datetime


def topics_over_time_analysis(
    topic_model: BERTopic,
    docs: list[str],
    topics: list[int],
    timestamps: list,  # e.g. datetime or numeric time bins
    nr_bins: int = 20,
    datetime_format: str = None
) -> Tuple[pd.DataFrame, Any]:
    """
    Compute topics over time and return the result and visualization.
    
    Args:
        topic_model: The fitted BERTopic model
        docs: List of documents
        topics: List of topic assignments for each document
        timestamps: List of timestamps (datetime objects, strings, or numeric)
        nr_bins: Number of time bins to create
        datetime_format: Format string for parsing datetime strings (if timestamps are strings)
        
    Returns:
        tuple: Topics over time DataFrame and visualization figure
    """
    # Convert string timestamps to datetime if needed
    if datetime_format and isinstance(timestamps[0], str):
        timestamps = [datetime.strptime(ts, datetime_format) for ts in timestamps]
    
    # Compute topics over time
    topics_time = topic_model.topics_over_time(
        docs, topics, timestamps, nr_bins=nr_bins
    )
    
    # Create visualization
    fig = topic_model.visualize_topics_over_time(topics_time)
    
    return topics_time, fig


def analyze_topic_trends(
    topics_time: pd.DataFrame,
    topic_id: int = None,
    top_n: int = 5
) -> pd.DataFrame:
    """
    Analyze trends for specific topics or top topics over time.
    
    Args:
        topics_time: DataFrame from topics_over_time_analysis
        topic_id: Specific topic ID to analyze (if None, analyzes top topics)
        top_n: Number of top topics to analyze if topic_id is None
        
    Returns:
        DataFrame with topic trends
    """
    if topic_id is not None:
        # Analyze specific topic
        topic_data = topics_time[topics_time['Topic'] == topic_id]
        return topic_data.sort_values('Timestamp')
    else:
        # Analyze top topics by frequency
        topic_counts = topics_time['Topic'].value_counts()
        top_topics = topic_counts.head(top_n).index
        
        return topics_time[topics_time['Topic'].isin(top_topics)].sort_values(['Topic', 'Timestamp'])


def get_topic_evolution(
    topics_time: pd.DataFrame,
    topic_id: int
) -> dict:
    """
    Get detailed evolution information for a specific topic.
    
    Args:
        topics_time: DataFrame from topics_over_time_analysis
        topic_id: Topic ID to analyze
        
    Returns:
        Dictionary with topic evolution statistics
    """
    topic_data = topics_time[topics_time['Topic'] == topic_id].sort_values('Timestamp')
    
    if len(topic_data) == 0:
        return {"error": f"Topic {topic_id} not found in time series data"}
    
    evolution = {
        "topic_id": topic_id,
        "total_periods": len(topic_data),
        "first_appearance": topic_data.iloc[0]['Timestamp'],
        "last_appearance": topic_data.iloc[-1]['Timestamp'],
        "peak_frequency": topic_data['Frequency'].max(),
        "peak_timestamp": topic_data.loc[topic_data['Frequency'].idxmax(), 'Timestamp'],
        "average_frequency": topic_data['Frequency'].mean(),
        "trend": "increasing" if topic_data['Frequency'].iloc[-1] > topic_data['Frequency'].iloc[0] else "decreasing"
    }
    
    return evolution


if __name__ == "__main__":
    from topic_model_fit_transform import topic_model_fit_transform
    
    # Sample documents with timestamps
    docs = [
        "Machine learning and artificial intelligence are revolutionizing technology.",
        "Data science involves statistics, programming, and domain expertise.",
        "COVID-19 pandemic has changed global health and economy.",
        "Vaccines and medical research are crucial for public health.",
        "Quantum computing could break current encryption methods.",
        "Cryptocurrency and blockchain technology are emerging trends.",
        "Climate change is affecting weather patterns worldwide.",
        "Renewable energy sources like solar and wind are growing.",
        "Stock market volatility affects investor confidence.",
        "Economic policies influence inflation and employment rates.",
        "Deep learning neural networks require large datasets.",
        "Natural language processing is advancing rapidly.",
        "Computer vision applications are expanding in healthcare.",
        "Robotics and automation are transforming manufacturing.",
        "Internet of Things devices are becoming more prevalent.",
        "Machine learning models are being deployed in production systems.",
        "Data privacy regulations are becoming more stringent.",
        "Edge computing is bringing AI closer to devices.",
        "Explainable AI is gaining importance in critical applications.",
        "Federated learning allows training without sharing raw data."
    ]
    
    # Create timestamps spanning multiple years
    timestamps = [
        "2020-01-15", "2020-03-20", "2020-06-10", "2020-09-05",
        "2021-02-14", "2021-05-08", "2021-08-12", "2021-11-30",
        "2022-01-20", "2022-04-15", "2022-07-22", "2022-10-18",
        "2023-01-10", "2023-03-25", "2023-06-15", "2023-09-08",
        "2023-12-01", "2024-02-14", "2024-05-20", "2024-08-10"
    ]
    
    print("Fitting BERTopic model...")
    model, topics, probs = topic_model_fit_transform(docs, calculate_probabilities=True)
    
    print("Analyzing topics over time...")
    topics_time, fig = topics_over_time_analysis(
        model, docs, topics, timestamps, 
        datetime_format="%Y-%m-%d"
    )
    
    print("Topics over time data:")
    print(topics_time.head(10))
    
    print("\nAnalyzing topic trends...")
    trends = analyze_topic_trends(topics_time, top_n=3)
    print("Top 3 topic trends:")
    print(trends)
    
    # Analyze evolution of first topic
    if len(topics_time) > 0:
        first_topic = topics_time['Topic'].iloc[0]
        evolution = get_topic_evolution(topics_time, first_topic)
        print(f"\nEvolution of topic {first_topic}:")
        for key, value in evolution.items():
            print(f"  {key}: {value}")
    
    # Show the visualization
    print("\nShowing topics over time visualization...")
    fig.show()
