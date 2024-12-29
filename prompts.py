##################################################
# prompts.py
##################################################
from textwrap import dedent

def get_summarization_prompt(user_queries):
    """
    Build a prompt for summarizing the intent/class from a set of user queries.
    """
    prompt = f"""\
    Review the following user queries and provide a summary of the intent. 
    Keep the summary generic and avoid referencing any named entities that appear in the queries.
    Queries:
    {user_queries}
    Summary:
    """
    return dedent(prompt)

def get_tag_generation_prompt(user_query, example_prompts=None):
    """
    Build a prompt for generating tags from a given user query.
    'example_prompts' can contain few-shot examples for better tagging performance.
    """
    examples_str = ""
    if example_prompts:
        examples_str = "\n".join([
            f"Query: \"{ex['query']}\"\nTags: {ex['tags']}" 
            for ex in example_prompts
        ])

    prompt = f"""\
    Generate descriptive tags for the following query. 
    Focus on user intention, relevant entities, and keywords.
    Extend these tags to related, unmentioned terms that are contextually relevant.

    Guidelines:
    1. Topic: Identify user intention or subject area the query pertains to.
    2. Entities: Recognize relevant or commonly used entities.
    3. Keywords: Extract key terms or verbs that define the query's intent.
    4. Related Tags: Include tags that are implied or contextually relevant, even if not explicitly mentioned.

    Example(s):
    {examples_str}

    Query: "{user_query}"
    Tags:
    """
    return dedent(prompt)

def get_prediction_prompt(user_query, retrieved_classes, class_summaries):
    """
    Build the final classification prompt using top-k class summaries.
    """
    classes_str = ", ".join(retrieved_classes)
    # Combine all relevant classes and their summaries
    # We’ll format them like:
    #   class_name: <summary>
    #   class_name: <summary>
    summaries_str = ""
    for c in retrieved_classes:
        summaries_str += f"{c}:\n{class_summaries[c].strip()}\n\n"

    prompt = f"""\
    Based on the given query, classify the user's intent into one of the following categories:
    {classes_str}

    {summaries_str}
    Query: "{user_query}"
    Class:
    """
    return dedent(prompt)
