"""
GLiFormer Base v1 Examples
==========================

Comprehensive examples demonstrating the capabilities of knowledgator/gliformer-base-v1,
a generalist multitask transformer encoder for information extraction.

Installation:
    pip install gliformer

Optional dependencies:
    pip install "gliformer[pdf]"  # For PDF layout processing
    pip install "gliformer[flash]" # For CUDA attention kernels

Usage:
    python gliformer_examples.py
"""

import json
from typing import Any, List

import torch
from pydantic import BaseModel


def setup_model() -> Any:
    """
    Load and initialize the GLiFormer Base v1 model.

    Returns:
        GLiFormer model instance ready for inference
    """
    from gliformer import GLiFormer

    print("Loading GLiFormer Base v1 model...")
    model = GLiFormer.from_pretrained(
        "knowledgator/gliformer-base-v1",
        load_tokenizer=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    print(f"Model loaded on {device}")

    return model


def example_named_entity_recognition(model: Any) -> None:
    """
    Demonstrate Named Entity Recognition (NER) with dynamic labels.

    Args:
        model: Loaded GLiFormer model
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Named Entity Recognition")
    print("=" * 60)

    text = "Alice works at Acme Corporation in London, UK."
    labels = ["person", "organization", "location"]

    print(f"\nInput text: {text}")
    print(f"Entity types: {labels}\n")

    entities = model.predict_entities(
        text,
        labels,
        threshold=0.5,
    )

    print("Extracted entities:")
    for entity in entities:
        print(
            f"  - {entity['text']:20s} => {entity['label']:15s} (score: {entity['score']:.3f})"
        )
        print(f"    Position: [{entity['start']}:{entity['end']}]")

    # Batched inference example
    print("\n--- Batched Inference ---")
    texts = [
        "Marie Curie worked at the University of Paris.",
        "Elon Musk founded SpaceX in California.",
    ]

    entities_batch = model.predict_entities(
        texts,
        ["person", "organization", "location"],
        batch_size=8,
    )

    for i, (text, ents) in enumerate(zip(texts, entities_batch)):
        print(f"\nText {i + 1}: {text}")
        for ent in ents:
            print(f"  {ent['text']} => {ent['label']}")


def example_text_classification(model: Any) -> None:
    """
    Demonstrate text classification with dynamic labels.

    Args:
        model: Loaded GLiFormer model
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Text Classification")
    print("=" * 60)

    text = "The new search feature is incredibly fast and easy to use!"
    labels = ["positive", "negative", "neutral"]

    print(f"\nInput text: {text}")
    print(f"Labels: {labels}\n")

    predictions = model.classify(
        text,
        labels,
        threshold=0.5,
    )

    print("Classification results:")
    for pred in predictions:
        print(f"  - {pred['class_name']:15s} (score: {pred['score']:.3f})")

    # Multi-task classification with named groups
    print("\n--- Multi-Task Classification ---")
    text2 = "Apple released a revolutionary new iPhone today."

    # Use inference directly for better handling of named groups
    results = model.inference(
        text2,
        classes={
            "sentiment": ["positive", "negative"],
            "topic": ["technology", "sports", "politics", "business"],
        },
        threshold=0.5,
    )

    print(f"\nInput: {text2}")

    # Results structure: {"classification": [ [sentiment_preds, topic_preds] ]}
    if "classification" in results and results["classification"]:
        class_results = results["classification"][0]  # List of group results

        # Map back to group names based on order
        group_names = ["sentiment", "topic"]

        for i, group_name in enumerate(group_names):
            if i < len(class_results):
                preds = class_results[i]
                print(f"\n{group_name.capitalize()}:")
                for pred in preds:
                    print(f"  - {pred['class_name']:15s} (score: {pred['score']:.3f})")


def example_relation_extraction(model: Any) -> None:
    """
    Demonstrate joint relation extraction.

    Args:
        model: Loaded GLiFormer model
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Joint Relation Extraction")
    print("=" * 60)

    text = "Alice works at Acme Corporation and lives in London."

    print(f"\nInput text: {text}\n")

    results = model.inference(
        text,
        joint_relations={
            "employment": {
                "entities": ["person", "organization"],
                "relations": ["works_at"],
            },
            "residence": {
                "entities": ["person", "location"],
                "relations": ["lives_in"],
            },
        },
        threshold=0.5,
    )

    print("Extracted relations:")
    if "joint_relex" in results and results["joint_relex"]:
        for relation in results["joint_relex"][0]:
            head_text = relation["head"]["text"]
            tail_text = relation["tail"]["text"]
            rel_type = relation["relation"]
            score = relation.get("score", 0)
            print(
                f"  {head_text:15s} => {rel_type:15s} => {tail_text:15s} (score: {score:.3f})"
            )
    else:
        print("  No relations extracted.")


def example_structured_extraction(model: Any) -> None:
    """
    Demonstrate multi-level structured extraction using Pydantic schemas.

    Args:
        model: Loaded GLiFormer model
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Structured Extraction (Multi-Level)")
    print("=" * 60)

    # Define nested schema
    class Employee(BaseModel):
        name: str
        role: str

    class Department(BaseModel):
        name: str
        employees: List[Employee]

    class Company(BaseModel):
        name: str
        departments: List[Department]

    text = (
        "At Acme Corporation, the Engineering department includes Alice, "
        "a software engineer, and Bob, a designer. The Sales department "
        "includes Carol, an account manager, and David, a sales director."
    )

    print(f"\nInput text:\n{text}\n")
    print("Schema: Company -> Departments -> Employees\n")

    records = model.structure(
        text,
        {"company": Company},
        validate_output=True,
    )

    print("Extracted structure:")
    print(json.dumps(records, indent=2))


def example_simple_structuring(model: Any) -> None:
    """
    Demonstrate simple structured extraction without Pydantic.

    Args:
        model: Loaded GLiFormer model
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Simple Structured Extraction")
    print("=" * 60)

    text = "Alice works at Acme Corporation as a software engineer."

    print(f"\nInput text: {text}\n")

    records = model.structure(
        text,
        {"employee": ["name", "company", "role"]},
    )

    print("Extracted records:")
    print(json.dumps(records, indent=2))


def example_multi_task_inference(model: Any) -> None:
    """
    Demonstrate running multiple tasks simultaneously.

    Args:
        model: Loaded GLiFormer model
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Multi-Task Inference")
    print("=" * 60)

    text = (
        "Alice joined Acme Corporation as a software engineer in the technology sector."
    )

    print(f"\nInput text: {text}\n")

    # Use a lower threshold to ensure we get some classification results
    # or use labels that are more distinctly activated by this text
    results = model.inference(
        text,
        entities=["person", "organization", "role"],
        classes=["business", "sports", "technology"],
        structures={"employee": ["name", "company"]},
        threshold=0.1,  # Lowered from 0.5 to capture weaker signals
    )

    print("Results by task:")

    if "ner" in results and results["ner"]:
        print("\n1. Named Entities:")
        for entity in results["ner"][0]:
            print(f"   - {entity['text']:20s} => {entity['label']}")

    if "classification" in results and results["classification"]:
        print("\n2. Classification:")
        class_data = results["classification"][0]

        # Handle potential nested structure depending on how classes were passed
        # When passing a list, it's treated as one group.
        if class_data:
            for pred in class_data:
                if isinstance(pred, dict):
                    print(f"   - {pred['class_name']:15s} (score: {pred['score']:.3f})")
                elif isinstance(pred, list):
                    # If it returns nested groups (unlikely for simple list input but possible)
                    for sub_pred in pred:
                        if isinstance(sub_pred, dict):
                            print(
                                f"   - {sub_pred['class_name']:15s} (score: {sub_pred['score']:.3f})"
                            )
        else:
            print("   No classifications above threshold.")

    if "structuring" in results and results["structuring"]:
        print("\n3. Structured Records:")
        print(f"   {json.dumps(results['structuring'][0], indent=6)}")


def example_text_embeddings(model: Any) -> None:
    """
    Demonstrate text embedding generation for semantic similarity.

    Args:
        model: Loaded GLiFormer model
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Text Embeddings & Semantic Similarity")
    print("=" * 60)

    texts = [
        "A scientist is working in a laboratory.",
        "A researcher is conducting an experiment.",
        "The cat is sleeping on the couch.",
    ]

    print("\nGenerating embeddings for:")
    for i, text in enumerate(texts, 1):
        print(f"  {i}. {text}")

    embeddings = model.embed_text(texts)

    print(f"\nEmbedding shape: {embeddings.shape}")
    print(f"Embedding dimension: {embeddings.shape[1]}")

    # Calculate pairwise cosine similarities
    import torch.nn.functional as F

    print("\nPairwise Cosine Similarities:")
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            sim = F.cosine_similarity(
                embeddings[i : i + 1], embeddings[j : j + 1]
            ).item()
            print(f"  Text {i + 1} vs Text {j + 1}: {sim:.4f}")


def example_pdf_layout_processing(model: Any) -> None:
    """
    Demonstrate PDF layout-aware entity extraction.
    Note: Requires 'pip install gliformer[pdf]' and a sample PDF file.

    Args:
        model: Loaded GLiFormer model
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 8: PDF Layout Processing")
    print("=" * 60)

    try:
        import os

        # Check if we have a sample PDF (this is illustrative)
        pdf_path = "sample_invoice.pdf"

        if not os.path.exists(pdf_path):
            print(f"\nNote: '{pdf_path}' not found. This example requires a PDF file.")
            print("To test this functionality:")
            print("  1. Install PDF dependencies: pip install 'gliformer[pdf]'")
            print("  2. Place a PDF file as 'sample_invoice.pdf'")
            print("  3. Run this example again")
            return

        print(f"\nProcessing PDF: {pdf_path}")

        results = model.parse_pdf(
            pdf_path,
            entities=["invoice number", "invoice date", "vendor", "total amount"],
            add_image_token=False,
            return_pixel_values=False,
            split_pages=True,
            return_pages=True,
            threshold=0.5,
        )

        print("\nExtracted entities by page:")
        for page_index, entities in zip(results["pages"], results["ner"]):
            print(f"\nPage {page_index + 1}:")
            if entities:
                for entity in entities:
                    print(f"  - {entity['text']:30s} => {entity['label']}")
            else:
                print("  No entities found")

    except ImportError:
        print(
            "\nPDF processing not available. Install with: pip install 'gliformer[pdf]'"
        )
    except Exception as e:
        print(f"\nError processing PDF: {e}")


def example_typed_fields_extraction(model: Any) -> None:
    """
    Demonstrate typed field extraction (dates, numbers, etc.).

    Args:
        model: Loaded GLiFormer model
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 9: Typed Fields Extraction")
    print("=" * 60)

    text = "The product was posted on March 3, 2024 and costs $29.99."

    print(f"\nInput text: {text}\n")

    # Define schema with typed fields
    structures = {
        "$root": {
            "title": "str",
            "posted": {"$type": "text", "$format": "date"},
            "price": {"$type": "number", "$unit": "USD"},
            "category": {
                "$type": "class",
                "$options": ["electronics", "furniture", "toys"],
            },
        }
    }

    result = model.inference(
        [text],
        structures=structures,
        reference_date="2024-01-01",
        decoder_kwargs={"typed_details": True},
    )

    print("Extracted typed fields:")
    if "structuring" in result and result["structuring"]:
        print(json.dumps(result["structuring"][0], indent=2))
    else:
        print("  No structured data extracted (may require specific training)")


def example_schema_reuse(model: Any) -> None:
    """
    Demonstrate creating and reusing a GLiFormerSchema.

    Args:
        model: Loaded GLiFormer model
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 10: Schema Reuse with GLiFormerSchema")
    print("=" * 60)

    from gliformer import GLiFormerSchema

    # Create reusable schema
    schema = GLiFormerSchema(
        entities=["person", "organization", "location"],
        classes=["positive", "negative", "neutral"],
        structures={"mention": ["name", "context"]},
    )

    texts = [
        "John Smith visited Google headquarters in Mountain View.",
        "Sarah Johnson gave a positive review about Microsoft.",
    ]

    print("\nProcessing multiple texts with same schema:")
    for text in texts:
        print(f"\nText: {text}")

        results = model.inference_from_schema(
            [text],
            schema,
            threshold=0.5,
        )

        if results.get("ner"):
            print("  Entities:", [(e["text"], e["label"]) for e in results["ner"][0]])

        if results.get("classification"):
            # results["classification"] is List[List[dict]] -> [[group1_preds, group2_preds...]]
            # For a single group schema, it's [[preds]]
            class_data = results["classification"][0]

            # Handle potential nested list structure if multiple groups or unexpected wrapping
            preds_to_print = []
            if class_data:
                if isinstance(class_data[0], list):
                    # Multiple groups: class_data is [group1_preds, group2_preds]
                    # Each group_preds is a list of dicts
                    for i, group_preds in enumerate(class_data):
                        preds_to_print.extend(
                            [
                                (p["class_name"], p["score"])
                                for p in group_preds
                                if isinstance(p, dict)
                            ]
                        )
                else:
                    # Single group: class_data is [pred1, pred2]
                    preds_to_print = [
                        (c["class_name"], c["score"])
                        for c in class_data
                        if isinstance(c, dict)
                    ]

            if preds_to_print:
                print("  Classes:", preds_to_print)
            else:
                print("  Classes: No predictions above threshold")


def main():
    """
    Main function to run all GLiFormer examples.
    """
    print("=" * 60)
    print("GLiFormer Base v1 - Comprehensive Examples")
    print("=" * 60)
    print("\nModel: knowledgator/gliformer-base-v1")
    print("Parameters: 264.2M")
    print("Tasks: NER, Classification, Relations, Structuring, Embeddings")
    print("=" * 60)

    # Setup model
    model = setup_model()

    # Run examples
    try:
        example_named_entity_recognition(model)
        example_text_classification(model)
        example_relation_extraction(model)
        example_structured_extraction(model)
        example_simple_structuring(model)
        example_multi_task_inference(model)
        example_text_embeddings(model)
        example_pdf_layout_processing(model)
        example_typed_fields_extraction(model)
        example_schema_reuse(model)

        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
