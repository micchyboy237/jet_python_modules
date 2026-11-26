# ─── Sample Documents ────────────────────────────────────────────────────────
# A diverse set of documents covering multiple themes with some outliers.


documents: list[str] = [
    # --- Cluster 1: Machine Learning & AI ---
    "Deep learning models have revolutionized computer vision tasks such as image classification and object detection.",
    "Transformer architectures like BERT and GPT have become the foundation of modern natural language processing systems.",
    "Reinforcement learning agents can learn optimal policies through trial and error in simulated environments.",
    "Convolutional neural networks excel at extracting spatial hierarchies of features from pixel data.",
    "Transfer learning allows pre-trained models to be fine-tuned on smaller domain-specific datasets efficiently.",
    "Generative adversarial networks pit a generator against a discriminator to produce realistic synthetic data.",
    "Attention mechanisms enable models to weigh the importance of different input tokens dynamically.",
    "Gradient descent optimization algorithms iteratively adjust model weights to minimize loss functions.",
    # --- Cluster 2: Climate Change & Environment ---
    "Rising global temperatures are causing polar ice caps to melt at an unprecedented rate.",
    "Deforestation in the Amazon rainforest contributes significantly to carbon dioxide emissions worldwide.",
    "Renewable energy sources like solar and wind power are becoming cost-competitive with fossil fuels.",
    "Ocean acidification threatens marine ecosystems by reducing the availability of carbonate ions for shell formation.",
    "Extreme weather events such as hurricanes and wildfires are increasing in frequency and intensity due to climate change.",
    "Carbon capture and storage technologies aim to remove CO2 from industrial emissions before it enters the atmosphere.",
    "Biodiversity loss is accelerating as habitats shrink and species face extinction pressures.",
    "The Paris Agreement sets international targets for limiting global warming to well below 2 degrees Celsius.",
    # --- Cluster 3: Web Development & JavaScript ---
    "React components use a virtual DOM to efficiently update the user interface when state changes.",
    "Node.js enables server-side JavaScript execution, allowing full-stack development with a single language.",
    "CSS Grid and Flexbox provide powerful layout systems for building responsive web designs.",
    "TypeScript adds static type checking to JavaScript, catching errors at compile time rather than runtime.",
    "Webpack bundles JavaScript modules and assets into optimized files for production deployment.",
    "RESTful APIs follow standard HTTP methods like GET, POST, PUT, and DELETE for resource manipulation.",
    "Single-page applications load a single HTML page and dynamically update content via AJAX requests.",
    "npm is the default package manager for the JavaScript runtime environment Node.js.",
    # --- Cluster 4: Ancient History & Archaeology ---
    "The Great Pyramid of Giza was constructed during the Fourth Dynasty of ancient Egypt around 2560 BCE.",
    "Roman aqueducts used gravity-fed channels to transport water across vast distances to urban centers.",
    "Cuneiform writing developed in Mesopotamia is considered one of the earliest known systems of written communication.",
    "The Terracotta Army was buried with Qin Shi Huang, the first emperor of China, to protect him in the afterlife.",
    "Stonehenge remains one of the most famous prehistoric monuments, with its purpose still debated by archaeologists.",
    "The Library of Alexandria was a major center of scholarship in the Hellenistic world before its destruction.",
    "Machu Picchu showcases the architectural ingenuity of the Inca civilization high in the Andes Mountains.",
    "Hieroglyphics were deciphered in the 19th century using the Rosetta Stone as a key translation tool.",
    # --- Cluster 5: Nutrition & Health ---
    "Omega-3 fatty acids found in fish support cardiovascular health and reduce inflammation in the body.",
    "Probiotics introduce beneficial bacteria into the gut microbiome, aiding digestion and immune function.",
    "Intermittent fasting restricts eating to specific time windows, potentially improving metabolic health.",
    "Vitamin D deficiency is linked to weakened bone density and increased risk of osteoporosis.",
    "Antioxidants neutralize free radicals that can cause cellular damage and contribute to aging.",
    "The Mediterranean diet emphasizes whole grains, olive oil, fruits, vegetables, and lean proteins.",
    "Hydration is essential for maintaining blood volume, regulating body temperature, and supporting kidney function.",
    "Excessive sugar consumption is associated with insulin resistance and an increased risk of type 2 diabetes.",
    # --- Outliers (should not fit cleanly into any main cluster) ---
    "The stock market experienced significant volatility following the unexpected announcement of interest rate changes.",
    "Quantum entanglement describes a phenomenon where particles remain correlated regardless of the distance separating them.",
    "Impressionist painters like Monet and Renoir focused on capturing the effects of light and movement in their works.",
    "The blockchain ledger provides a decentralized and tamper-resistant record of cryptocurrency transactions.",
]


if __name__ == "__main__":
    from jet.adapters.llama_cpp.cluster_utils.clusterer_llm import (
        run_clustering_pipeline,
    )

    # Run the pipeline
    granularity = "balanced"
    clustered = run_clustering_pipeline(documents, granularity=granularity)

    # Print results
    print("\n" + "=" * 80)
    print("DOCUMENT CLUSTERING RESULTS")
    print("=" * 80)

    for cluster_id, info in sorted(clustered.items()):
        print(f"\n📁 Cluster {cluster_id}: {info['topic']}")
        print(f"   Documents ({len(info['document_indices'])}):")
        for idx in info["document_indices"]:
            preview = (
                documents[idx][:100] + "..."
                if len(documents[idx]) > 100
                else documents[idx]
            )
            print(f"     [{idx}] {preview}")

    print("\n" + "=" * 80)
