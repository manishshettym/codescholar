from tqdm import tqdm
import openai
import numpy as np

def embed_programs_gpt(args, progs):
    embeddings = []
    for prog in tqdm(progs, desc="[embed]"):
        response = openai.Embedding.create(input = [prog], model="text-embedding-ada-002")
        prog_embedding = np.array(response['data'][0]['embedding'])
        prog_embedding = prog_embedding.reshape(1, -1)
        embeddings.append(prog_embedding)

    return embeddings
