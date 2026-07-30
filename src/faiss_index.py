import numpy as np
import faiss

EMBEDDINGS_PATH = "data/embeddings.npy"
INDEX_PATH = "data/scifact.index"

embeds = np.load(EMBEDDINGS_PATH).astype(np.float32)
faiss.normalize_L2(embeds)

faiss_index = faiss.IndexFlatIP(embeds.shape[1])
faiss_index.add(embeds)

faiss.write_index(faiss_index, INDEX_PATH)
