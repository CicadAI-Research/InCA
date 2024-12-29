##################################################
# inca.py
##################################################

import subprocess
import json
import numpy as np
from numpy.linalg import inv
from sentence_transformers import SentenceTransformer

from prompts import (
    get_summarization_prompt, 
    get_tag_generation_prompt,
    get_prediction_prompt
)

class OllamaLLM:
    """
    A simple interface to call a local LLM using Ollama CLI.
    Adjust 'model_name' or the command flags as needed.
    """
    def __init__(self, model_name: str = "mistral"):
        self.model_name = model_name

    def generate(self, prompt: str, num_ctx_tokens=2048) -> str:
        """
        Calls Ollama with the given prompt and returns the generated text.
        """
        # You can add or adjust arguments to 'ollama generate' as needed
        cmd = [
            "ollama",
            "generate",
            "-m", self.model_name,
            "-p", prompt,
            "--num_ctx", str(num_ctx_tokens)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.stdout.strip()


class ExternalContinualLearner:
    """
    The External Continual Learner (ECL) that incrementally maintains:
      - A mean embedding vector for each class
      - A shared covariance matrix across all classes
    and uses Mahalanobis distance to retrieve the top-k candidate classes.
    """
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.class_means = {}       # class_name -> mean vector
        self.class_counts = {}      # class_name -> total count of tags
        self.sum_outer_products = np.zeros((embedding_dim, embedding_dim))
        self.num_tags_total = 0     # total count of tags across all classes

        # We store a single covariance matrix for all classes:
        self.cov_matrix = np.eye(embedding_dim)

    def update_class_distribution(self, class_name, tag_embeddings):
        """
        Incrementally update the mean embedding for the given class
        and the shared covariance matrix.
        """
        # Convert to np array
        tag_embeddings = np.array(tag_embeddings)  # shape: (num_tags, dim)
        num_new_tags = tag_embeddings.shape[0]
        
        # -- Update class mean --
        if class_name not in self.class_means:
            # If this is the first time we're seeing this class
            self.class_means[class_name] = np.mean(tag_embeddings, axis=0)
            self.class_counts[class_name] = num_new_tags
        else:
            old_mean = self.class_means[class_name]
            old_count = self.class_counts[class_name]
            
            new_count = old_count + num_new_tags
            new_mean = (old_mean * old_count + np.sum(tag_embeddings, axis=0)) / new_count
            
            self.class_means[class_name] = new_mean
            self.class_counts[class_name] = new_count

        # -- Update shared covariance matrix statistics --
        # Sum of outer products for each new tag
        for emb in tag_embeddings:
            diff = emb - self.class_means[class_name]
            # We can store them for a full update once per class or do immediate partial updates
            # For simplicity, let's do immediate partial updates below:
            #   we add outer products of (emb - old overall mean?), etc.
            # 
            # However, truly correct incremental update of the covariance
            # in the presence of changing means can be non-trivial.
            # For a simplified approach, we keep a sum of (z)(z^T), then later
            # use E(Z Z^T) - E(Z) E(Z)^T approach. 
            pass
        
        # In practice, a simpler approach is to store all embeddings or store
        # partial sums to update the covariance in a single pass. We'll do it
        # with partial sums approach, for demonstration:
        self.num_tags_total += num_new_tags
        # Add sum of outer products of the new embeddings
        for emb in tag_embeddings:
            self.sum_outer_products += np.outer(emb, emb)

        # Recompute the covariance from the sums:
        # Cov = E[zz^T] - E[z] E[z]^T
        # E[z] = average of all tags across all classes
        # E[zz^T] = sum_outer_products / num_tags_total
        # For large data, you might want to store total mean of all tags across classes
        # This is a simplified approach for demonstration:

        # total mean of all tags across all classes
        global_mean = self._compute_global_mean()
        # E[zz^T]
        e_zzT = self.sum_outer_products / self.num_tags_total
        # E[z]E[z]^T
        e_z_e_zT = np.outer(global_mean, global_mean)
        # get the new covariance
        new_cov = e_zzT - e_z_e_zT
        # Add a small regularization to avoid singular matrix
        new_cov += np.eye(self.embedding_dim) * 1e-6
        self.cov_matrix = new_cov

    def _compute_global_mean(self):
        """
        Return the global mean of *all tags* across *all classes*.
        We can get this from the class means times their frequencies.
        """
        if self.num_tags_total == 0:
            return np.zeros((self.embedding_dim,))
        weighted_sum = np.zeros((self.embedding_dim,))
        for c, mean_vec in self.class_means.items():
            count = self.class_counts[c]
            weighted_sum += mean_vec * count
        return weighted_sum / self.num_tags_total

    def retrieve_top_k(self, tag_embeddings, k=3):
        """
        For each test instance, compute the average Mahalanobis distance
        to each class distribution, then pick the top k classes.
        """
        # Convert to np array
        tag_embeddings = np.array(tag_embeddings)
        if len(tag_embeddings.shape) == 1:
            tag_embeddings = tag_embeddings[None, :]

        # Average embedding of the input’s tags
        # (Though in the paper, the distance is the average of distances, but
        #  it’s effectively the same to just compare to the average embedding.)
        avg_emb = np.mean(tag_embeddings, axis=0)  # shape: (dim,)

        # Mahalanobis distance: d^2 = (z - mu)^T Sigma^{-1} (z - mu)
        # We'll do a single inverse once
        inv_cov = inv(self.cov_matrix)

        class_distances = []
        for c, mean_vec in self.class_means.items():
            diff = avg_emb - mean_vec
            mahal = diff.T @ inv_cov @ diff  # scalar
            # Use sqrt only if you want the actual distance
            # but if we only compare them, the ordering is the same with or without sqrt.
            distance = np.sqrt(mahal)
            class_distances.append((c, distance))

        # sort by ascending distance
        class_distances.sort(key=lambda x: x[1])
        return [cd[0] for cd in class_distances[:k]]


class InCA:
    """
    Main InCA pipeline that uses:
      1) An LLM to generate class summaries (once per class).
      2) An External Continual Learner (ECL) to store class distributions.
      3) The same LLM for tag generation and final classification.
    """
    def __init__(self, llm_model_name="mistral", embed_model_name="sentence-transformers/paraphrase-MiniLM-L6-v2"):
        self.llm = OllamaLLM(model_name=llm_model_name)
        self.embed_model = SentenceTransformer(embed_model_name)
        # Dimensionality of the embedding model:
        self.embedding_dim = self.embed_model.get_sentence_embedding_dimension()

        self.ecl = ExternalContinualLearner(embedding_dim=self.embedding_dim)
        self.class_summaries = {}  # class_name -> summary (text)

    def add_new_class(
        self, 
        class_name: str, 
        user_queries_for_class, 
        num_summary_examples=5
    ):
        """
        1) Generate a summary of this class (via a prompt).
        2) Generate tags for each example and update ECL distribution.
        """
        # 1) Summarize the class using a random subset (up to num_summary_examples)
        if len(user_queries_for_class) > num_summary_examples:
            chosen = user_queries_for_class[:num_summary_examples]
        else:
            chosen = user_queries_for_class
        
        # Build a multi-line bullet or newline list of queries
        joined_queries = "\n".join(f"- {q}" for q in chosen)
        summary_prompt = get_summarization_prompt(joined_queries)
        summary = self.llm.generate(summary_prompt, num_ctx_tokens=4096)
        self.class_summaries[class_name] = summary

        # 2) For each example of this class, generate tags and update ECL
        all_tag_embeddings = []
        for query in user_queries_for_class:
            tag_prompt = get_tag_generation_prompt(user_query=query)
            raw_tag_str = self.llm.generate(tag_prompt, num_ctx_tokens=4096)
            # Attempt a simple parse of lines or comma separated
            parsed_tags = self._parse_tags(raw_tag_str)
            # embed each tag
            tag_embs = self.embed_model.encode(parsed_tags)
            # store for this class
            all_tag_embeddings.extend(tag_embs)

        # Update ECL for this new class
        if len(all_tag_embeddings) > 0:
            self.ecl.update_class_distribution(class_name, all_tag_embeddings)

    def predict_class(self, query: str, k=3) -> str:
        """
        1) Generate tags for the query.
        2) Retrieve top-k classes from ECL.
        3) Prompt LLM with those classes' summaries for final classification.
        """
        # Generate tags
        tag_prompt = get_tag_generation_prompt(query)
        raw_tag_str = self.llm.generate(tag_prompt, num_ctx_tokens=4096)
        parsed_tags = self._parse_tags(raw_tag_str)
        # get embeddings
        tag_embs = self.embed_model.encode(parsed_tags)
        # retrieve top-k
        candidate_classes = self.ecl.retrieve_top_k(tag_embs, k=k)
        # final classification
        pred_prompt = get_prediction_prompt(query, candidate_classes, self.class_summaries)
        prediction = self.llm.generate(pred_prompt, num_ctx_tokens=4096)

        # The LLM might produce additional text. 
        # We'll parse until we see a newline or punctuation. Adjust as needed:
        prediction_line = prediction.strip().split("\n")[0]
        # Or parse out something like "Class: <class>"
        # We'll do a simple parse:
        predicted_label = prediction_line.strip()
        return predicted_label

    def _parse_tags(self, raw_tag_str: str):
        """
        Given the raw text from the LLM for the 'Tags:' prompt, 
        parse it into a list of individual tags. This can be as 
        simple or as sophisticated as you need.
        """
        # Example raw_tag_str:
        # "airline fees, carry-on, delta airlines, travel, pay, luggage"
        # We do a quick split by commas or newlines
        # Also strip out any bullet points or formatting
        possible_separators = [",", "\n", ";"]
        for sep in possible_separators:
            if sep in raw_tag_str:
                items = [t.strip("-* \t") for t in raw_tag_str.split(sep)]
                tags = [i.strip() for i in items if i.strip()]
                if len(tags) > 1:
                    return tags
        # fallback: single line
        tags = raw_tag_str.strip().split()
        return tags

