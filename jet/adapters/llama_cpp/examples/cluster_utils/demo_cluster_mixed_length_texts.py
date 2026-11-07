from jet.adapters.llama_cpp.cluster_utils import TextsClusterer

if __name__ == "__main__":
    mixed_texts = [
        "machine learning",
        "AI technology",
        "deep learning",
        "neural networks",
        "The product quality exceeded my expectations significantly.",
        "Customer service was unresponsive and quite disappointing.",
        "Fast delivery and excellent packaging throughout.",
        """Cloud computing provides scalable resources on-demand. Major providers like AWS and Azure offer comprehensive services
for businesses of all sizes. This technology enables cost-effective infrastructure management.""",
        """Web development combines frontend and backend technologies. Modern frameworks enable rapid development of responsive
applications. Full-stack developers work across the entire technology stack.""",
        """Artificial intelligence is transforming industries worldwide. Machine learning algorithms can now diagnose diseases,
drive cars, and translate languages with remarkable accuracy. Deep learning, a subset of AI, uses neural networks with
many layers to process complex patterns in data.
The ethical implications of AI development are increasingly important. Questions about bias in algorithms, job displacement,
and privacy concerns need careful consideration. Governments and organizations are developing guidelines for responsible AI use.
Future advancements in quantum computing may accelerate AI capabilities exponentially. Researchers are exploring new architectures
that could solve previously intractable problems in optimization and simulation.""",
        """Data science combines statistical analysis, programming, and domain expertise. Data scientists extract meaningful insights
from large datasets to inform business decisions. They use tools like Python, R, and SQL to clean, analyze, and visualize data.
The field continues to evolve with new techniques in deep learning and automated machine learning. Organizations are investing
heavily in data infrastructure and talent to remain competitive in the data-driven economy.""",
        "cybersecurity",
        "blockchain technology",
        "The app crashes frequently during peak hours.",
        "Great user interface but needs better documentation.",
        """Internet of Things connects devices globally. Smart homes automate lighting, temperature, and security.
Industrial IoT optimizes manufacturing processes through real-time monitoring.""",
        "Python programming",
        "JavaScript development",
        """DevOps bridges development and operations teams. Continuous integration and deployment automate software releases.
Containerization with Docker ensures consistent environments across development and production.""",
        "The battery life is terrible, lasting only 3 hours.",
        "Excellent value for money with premium features included.",
        # Added some distinct/outlier-like texts to demonstrate outlier detection
        "Sushi is best eaten fresh with soy sauce.",
        "The cat sat on the mat.",
        "Quantum entanglement defies classical intuition.",
        "Bananas are rich in potassium.",
        "The quick brown fox jumps over the lazy dog.",
    ]

    print("=" * 60)
    print("Mixed-Length Text Clustering Demo (Llama.cpp)")
    print("=" * 60)

    # Initialize with defaults from config
    clusterer = TextsClusterer(
        min_cluster_size=2,
        # max_tokens_safe=None will trigger dynamic calculation
    )

    results = clusterer.fit_predict(mixed_texts)
    clusters = clusterer.get_cluster_texts(mixed_texts)
    analysis = clusterer.get_detailed_analysis(mixed_texts)

    print("\n" + "=" * 60)
    print("DETAILED CLUSTER ANALYSIS:")
    print("=" * 60)

    # Separate regular clusters and noise for better display order
    regular_clusters = {k: v for k, v in analysis.items() if k != "Noise/Outliers"}
    noise_cluster = {k: v for k, v in analysis.items() if k == "Noise/Outliers"}

    # Print regular clusters first
    for cluster_name, info in sorted(regular_clusters.items()):
        print(f"\n{cluster_name}:")
        print(f"  Count: {info['count']}")
        print(f"  Length range: {info['min_length']} - {info['max_length']} words")
        print(f"  Average length: {info['avg_length']:.1f} words")
        print(f"  All texts:")
        for idx, (text, prob) in enumerate(
            zip(info["all_texts"], info["probabilities"]), 1
        ):
            preview = text[:100] + "..." if len(text) > 100 else text
            # Display probability score
            print(f"    {idx}. [{prob:.3f}] {preview}")

    # Print Noise/Outliers last if they exist
    if noise_cluster:
        for cluster_name, info in noise_cluster.items():
            print(f"\n{cluster_name}:")
            print(f"  Count: {info['count']}")
            print(f"  Length range: {info['min_length']} - {info['max_length']} words")
            print(f"  Average length: {info['avg_length']:.1f} words")
            print(f"  All texts:")
            for idx, (text, prob) in enumerate(
                zip(info["all_texts"], info["probabilities"]), 1
            ):
                preview = text[:100] + "..." if len(text) > 100 else text
                # Display probability score (will be 0.000 for noise)
                print(f"    {idx}. [{prob:.3f}] {preview}")

    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {results['n_clusters']} clusters, {results['n_noise']} outliers")
    print(f"{'=' * 60}")
