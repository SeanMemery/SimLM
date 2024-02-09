import os, json, sys
from pathlib import Path
from tqdm import tqdm
import numpy as np
from scipy.spatial.distance import cdist

sys.path.append(str(Path('.').absolute().parent))
from path import *
from src.llm import *

EXAMPLE_FILE = DATASTORE_DIR + "/example_store/examples.json"
EMBEDDING_FILE = DATASTORE_DIR + "/example_store/example_embeddings.npy"

SUMMARY_MODEL = "gpt-3.5-turbo"

class ExampleStore():
    def __init__(self, retrieval_only=True):
        self.examples, self.embeddings = self.load_examples()

        config = {
            "ctx_n": 4096,
            "temperature": 0.7,
            "top_k": 40,
            "top_p": 0.9,
            "num_predict": 256,
            "template": None,
        }
        embedding_model_name = "tinyllama" #TODO: programmatically get this
        self.embedding_llm = LLM(embedding_model_name, config)

        if not retrieval_only:
            self.summary_llm = LLM(SUMMARY_MODEL, config)

    def load_examples(self):
        """
        Load all examples from the datastore.
        """
        if not os.path.exists(EXAMPLE_FILE):
            json.dump([], open(EXAMPLE_FILE, "w"))
            examples = []
        else:
            examples = json.load(open(EXAMPLE_FILE))
        
        if not os.path.exists(EMBEDDING_FILE):
            # Save empty 2D array
            np.save(EMBEDDING_FILE, np.empty((0, 2048))) 
            embeddings = np.empty((0, 2048))
        else:
            embeddings = np.load(EMBEDDING_FILE)

        return examples, embeddings
 
    def exists(self, context_hash):
        """
        Check if an example exists in the datastore.
        """
        if len(self.examples) == 0:
            return False
        
        for example in self.examples:
            if example["context_hash"] == context_hash:
                return True
        return False

    def add_example(self, entry):
        """
        Add a single example to the datastore.
        """
        context_hash = len(entry["context"]) * len("".join(entry["events"]))
        if self.exists(context_hash):
            print(f"Example already exists for context {context_hash}")
            return
        
        # Summarise the context to a single paragraph
        summary = self.summarise(entry)
        embedding = self.embedding_llm.embedding(entry["task"]["prompt"])

        # Store the results in searchable format
        example = {
            "context_hash": context_hash,
            "summary": summary,
            "task_prompt": entry["task"]["prompt"],
            "raw": entry,
        }
                
        self.examples.append(example)
        self.embeddings = np.vstack([self.embeddings, embedding])  

        self.save_examples()

    def save_examples(self):
        """
        Save all examples to the datastore.
        Important to save both the embeddings and the summaries for each example. Embeddings are saved as a numpy array, summaries are saved in a JSON file.
        Embeddings should be searchable using a similarity search, returning the example context_hash value
        """

        np.save(EMBEDDING_FILE,  self.embeddings)
        json.dump(self.examples, open(EXAMPLE_FILE, "w"), indent=4)

    # Retrieve K examples from the datastore, using a combination of filters and similarity search
    def retrieve_examples(self, filters, task_prompt, top_k=3):
        """
        Retrieve examples from the datastore using a combination of filters and similarity search.
        """

        # Apply filters to examples, copy to avoid modifying original, remove filters that don't apply in both examples and embeddings
        filtered_examples = self.examples.copy()
        for key, value in filters.items():
            filtered_examples = [example for example in filtered_examples if example["raw"][key] == value]
        filtered_embeddings = self.embeddings.copy()[[index for index, example in enumerate(self.examples) if example in filtered_examples]]

        # Calculate cosine similarity
        query_embedding = self.embedding_llm.embedding(task_prompt)
        query_embedding = np.array(query_embedding).reshape(1, -1)
        similarities = 1 - cdist(query_embedding, filtered_embeddings, metric='cosine')
        # Get indices of top k similar embeddings
        top_k_indices = np.argsort(similarities[0])[::-1][:top_k]

        # Get examples from top_k_indices
        top_k_examples = []
        for index in top_k_indices:
            top_k_examples.append(filtered_examples[index]["summary"])

        return top_k_examples 

    def summarise(self, entry):
        """
        Summarise the context of a dataframe of results to a single paragraph.
        """

        starting_board_state = entry["board_states"][0]["text"]
        starting_board_state = f"#Starting Board State:\n{starting_board_state}\n\n"

        split = "In this case, you're task is:" # TODO: why you're ???
        to_summarise = split + entry["context"].split(split)[-1]
        
        summarise_system_prompt = f"""
Create an in depth summary of the following text, with all information, output as bullet points. Be sure to match the order of information: [REASONING, SIMULATION OUTPUT, CRITIQUE, SIMULATION OUTPUT, etc] and lose no information\n
"""
        summarise_prompt = f"""### Text to Summarise:\n
{ to_summarise }
\n\n
### Summary:\n"""

        self.summary_llm.set_system_prompt(summarise_system_prompt)
        summary = self.summary_llm.generate(summarise_prompt)
        self.summary_llm.reset()

        total_example = f"{starting_board_state}#Task:\n{entry['task']['prompt']}\n\n#Summary:\n{summary}"

        return total_example

def extract_successful_examples(df):
    """
    Extract successful examples from a dict of results.
    """
    if len(df) == 0:
        return df
    
    filtered_df = []
    
    for entry in df:
        if entry["passed"] == False or entry["method"] == "random" or len(entry["context"]) == 0:
            continue
        filtered_df.append(entry)

    return filtered_df

def load_from_folder(folder):
    """
    Load a json dict of results from a folder.
    """
    files = os.listdir(folder)
    df = []
    if len(files) == 0:
        return df
    for file in files:
        df += json.load(open(f"{folder}/{file}"))
    return df

def main():
    """
    Main function.
    """
    retrieval_only = False
    example_store = ExampleStore(retrieval_only)
    successful_tasks = []

    # Get all results in the results directory
    results_folders = os.listdir(RESULTS_DIR)
    for folder in results_folders:
        df = load_from_folder(f"{RESULTS_DIR}/{folder}")
        successful_tasks += extract_successful_examples(df)

    # Loop through each successful task
    for entry in tqdm(successful_tasks, desc="Storing examples"):
        example_store.add_example(entry)

if __name__ == "__main__":
    main()

