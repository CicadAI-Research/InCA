##################################################
# run_inca_example.py
##################################################

"""
A small usage example of the InCA pipeline.

In a real scenario, you will have your own data sets
split per class. For demonstration, we keep it simple.
"""

from inca import InCA

def main():
    # 1) Initialize the pipeline
    inca = InCA(
        llm_model_name="mistral",  # Adjust to your model's name used in Ollama
        embed_model_name="sentence-transformers/paraphrase-MiniLM-L6-v2"
    )

    # 2) Suppose we have data for three classes (toy example):
    class_a_data = [
        "How can I apply for a new credit card?",
        "I want to apply for a VISA card.",
        "Is this where I apply for a Mastercard?",
        "Steps for signing up for a new credit card?",
        "Am I eligible to open a new credit card?",
    ]
    class_b_data = [
        "When is my next payday?",
        "Could you tell me the date of my next paycheck?",
        "How often do we get paid?",
        "Can you confirm my next salary deposit?",
    ]
    class_c_data = [
        "Help me transfer money to a foreign bank.",
        "How do I send funds to an overseas account?",
        "Can I wire money abroad?",
        "I want to do an international money transfer."
    ]

    # 3) Incrementally add the classes (simulate class-incremental)
    inca.add_new_class("credit_card_application", class_a_data)
    inca.add_new_class("payday", class_b_data)
    inca.add_new_class("international_transfer", class_c_data)

    # 4) Try some queries for inference:
    queries = [
        "I need to figure out how to get a Mastercard",
        "When will I receive my salary next?",
        "I want to wire money overseas tomorrow"
    ]
    for q in queries:
        pred = inca.predict_class(q, k=2)
        print(f"Query: {q}\nPredicted: {pred}\n{'-'*40}")

if __name__ == "__main__":
    main()
