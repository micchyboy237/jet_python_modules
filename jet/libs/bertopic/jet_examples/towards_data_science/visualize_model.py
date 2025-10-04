from bertopic import BERTopic
from typing import Dict, Any


def visualize_model(
    topic_model: BERTopic, 
    topics_over_time=None, 
    probs=None, 
    doc_index: int = None,
    save_plots: bool = False,
    plot_path: str = "bertopic_plots"
) -> Dict[str, Any]:
    """
    Show several visualizations from the topic model.
    
    Args:
        topic_model: The fitted BERTopic model
        topics_over_time: Topics over time data (optional)
        probs: Topic probabilities for documents (optional)
        doc_index: Index of document to show distribution for (optional)
        save_plots: Whether to save plots to files
        plot_path: Directory to save plots to
        
    Returns:
        dict: Dictionary containing all generated plot figures
    """
    import os
    
    if save_plots and not os.path.exists(plot_path):
        os.makedirs(plot_path)
    
    plots = {}
    
    # 1. Intertopic distance map
    try:
        fig1 = topic_model.visualize_topics()
        plots["intertopic"] = fig1
        if save_plots:
            fig1.write_html(f"{plot_path}/intertopic_distance.html")
            fig1.write_image(f"{plot_path}/intertopic_distance.png")
        print("✓ Intertopic distance map created")
    except Exception as e:
        print(f"✗ Failed to create intertopic distance map: {e}")
        plots["intertopic"] = None
    
    # 2. Bar chart (top words per topic)
    try:
        fig2 = topic_model.visualize_barchart()
        plots["barchart"] = fig2
        if save_plots:
            fig2.write_html(f"{plot_path}/topic_barchart.html")
            fig2.write_image(f"{plot_path}/topic_barchart.png")
        print("✓ Topic bar chart created")
    except Exception as e:
        print(f"✗ Failed to create bar chart: {e}")
        plots["barchart"] = None
    
    # 3. Heatmap of topic similarities
    try:
        fig3 = topic_model.visualize_heatmap()
        plots["heatmap"] = fig3
        if save_plots:
            fig3.write_html(f"{plot_path}/topic_heatmap.html")
            fig3.write_image(f"{plot_path}/topic_heatmap.png")
        print("✓ Topic heatmap created")
    except Exception as e:
        print(f"✗ Failed to create heatmap: {e}")
        plots["heatmap"] = None
    
    # 4. Document topic distribution (if probabilities and doc index provided)
    if probs is not None and doc_index is not None:
        try:
            fig4 = topic_model.visualize_distribution(probs[doc_index])
            plots["doc_distribution"] = fig4
            if save_plots:
                fig4.write_html(f"{plot_path}/doc_{doc_index}_distribution.html")
                fig4.write_image(f"{plot_path}/doc_{doc_index}_distribution.png")
            print(f"✓ Document {doc_index} topic distribution created")
        except Exception as e:
            print(f"✗ Failed to create document distribution: {e}")
            plots["doc_distribution"] = None
    else:
        plots["doc_distribution"] = None
    
    # 5. Topics over time (if time series data provided)
    if topics_over_time is not None:
        try:
            fig5 = topic_model.visualize_topics_over_time(topics_over_time)
            plots["topics_over_time"] = fig5
            if save_plots:
                fig5.write_html(f"{plot_path}/topics_over_time.html")
                fig5.write_image(f"{plot_path}/topics_over_time.png")
            print("✓ Topics over time visualization created")
        except Exception as e:
            print(f"✗ Failed to create topics over time: {e}")
            plots["topics_over_time"] = None
    else:
        plots["topics_over_time"] = None
    
    return plots


def show_visualization(plots: Dict[str, Any], plot_name: str):
    """
    Display a specific visualization from the plots dictionary.
    
    Args:
        plots: Dictionary of plots from visualize_model
        plot_name: Name of the plot to show ('intertopic', 'barchart', 'heatmap', etc.)
    """
    if plot_name in plots and plots[plot_name] is not None:
        plots[plot_name].show()
    else:
        print(f"Plot '{plot_name}' not available or failed to create")


if __name__ == "__main__":
    from topic_model_fit_transform import topic_model_fit_transform
    from topics_over_time_analysis import topics_over_time_analysis
    
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
        "Internet of Things devices are becoming more prevalent."
    ]
    
    timestamps = [2020, 2020, 2020, 2020, 2021, 2021, 2021, 2021, 2022, 2022, 2022, 2022, 2023, 2023, 2023]
    
    # Fit the model
    print("Fitting BERTopic model...")
    model, topics, probs = topic_model_fit_transform(docs, calculate_probabilities=True)
    
    # Get topics over time
    print("Analyzing topics over time...")
    topics_time, _ = topics_over_time_analysis(model, docs, topics, timestamps)
    
    # Create all visualizations
    print("Creating visualizations...")
    plots = visualize_model(
        model, 
        topics_over_time=topics_time, 
        probs=probs, 
        doc_index=0,
        save_plots=True
    )
    
    # Show specific visualizations
    print("\nShowing intertopic distance map...")
    show_visualization(plots, "intertopic")
    
    print("\nShowing topic bar chart...")
    show_visualization(plots, "barchart")
    
    print("\nShowing topics over time...")
    show_visualization(plots, "topics_over_time")
