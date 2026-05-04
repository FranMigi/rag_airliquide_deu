import json
import chromadb # Base vectorielle open-source Chroma
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer # SBERT Embedding
from pathlib import Path

# Gére les chemins vers les docs source
PAGES_PATH = Path("data/processed/pages.jsonl")
CHROMA_DIR = Path("data/processed/chroma") # Le dossier chroma n'existe pas encore mais pas de pb car c'est juste un chemin pour le moment
COLLECTION_NAME = "airliquide_pages"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

print("Initialization OK")

#Transforme tous les objets JSON en liste Python contenant toutes les lignes sous forme de dict
def load_pages():
    pages = []
    with open(PAGES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            pages.append(json.loads(line))
    return pages

def main():

    # Sécurité si pas de fichier JSON trouvé
    if not PAGES_PATH.exists():
        raise FileNotFoundError("pages.jsonl not found. Run ingest.py first.")

    print("Loading pages...")
    pages = load_pages()
    print(f"{len(pages)} pages loaded.")

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Initializing Chroma DB...")
    # La je crée mon dossier chroma si besoin
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    #  Initialise un client Chroma qui sauvegarde les données sur disque de manière "Persistent" = les données restent stockées entre les exécutions
    # Stock la base de données : CHROMA_DIR 
    # setting pour accéder à la config. de Chroma
    # anonymized_telemetry=False => empeche l'envoi de statistique en externe, ajoute un peu de confidentialité (même si statistiques anonymes)
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False)
    )

    # Si la base vect. existe déjà, on la supprime (MVP simple), pas optimisé, on pourra juste modifier ce qui en a besoin
    try:
        client.delete_collection(COLLECTION_NAME)
    except:
        pass

    collection = client.create_collection(name=COLLECTION_NAME)

    print("Creating embeddings and indexing...")

    batch_size = 64

    for i in range(0, len(pages), batch_size):

        batch = pages[i:i+batch_size]

        texts = [p["text"] for p in batch]
        embeddings = model.encode(texts).tolist()

        ids = [f'{p["doc"]}::p{p["page"]}' for p in batch]
        metadatas = [{"doc": p["doc"], "page": p["page"]} for p in batch]

        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

        print(f"Indexed {min(i+batch_size, len(pages))}/{len(pages)}")

    print("Indexing complete.")

if __name__ == "__main__": #Pas obligatoire ici car j'import jamais le fichier par la suite mais bonne pratique
    main()
