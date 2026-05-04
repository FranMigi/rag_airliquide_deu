#J'utilise des print 
#tout au long du programme pour suivre le déroulement

print("START ingest.py")

#Pas donner d'alias à json et fitz car facile à écrire
import json 
import fitz  # PyMuPDF
from pathlib import Path

# Chemins
INPUT_FOLDER = Path("data/raw_pdf")
OUTPUT_FOLDER = Path("data/processed")  
OUTPUT_FILE = OUTPUT_FOLDER / "pages.jsonl"

# Créer le dossier de sortie s'il n'existe pas et le dossier mère aussi s'il n'existe pas
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

#Ici la boucle for n'est pas utile quand il y a 1 seul document
for pdf_path in INPUT_FOLDER.glob("*.pdf"):
    print(f"Traitement de {pdf_path.name}")

    try:
        # Ouvre le PDF
        doc = fitz.open(pdf_path)
        print(f"PDF ouvert avec succès : {len(doc)} pages")

        # Ouvre le fichier de sortie en mode ajout sous la norme UTF-8 
        # with ... as => close automatiquement
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            for page_number in range(len(doc)):
                page = doc[page_number]
                text = page.get_text()

                # Ignore les pages vides
                if not text.strip():
                    continue

                # Structure des données
                record = {
                    "doc": pdf_path.name, #on utilise .name pour garder que le nom du document
                    "page": page_number + 1,  # +1 car avec ordinateur page[1] => page[0] en langage informatique
                    "text": text
                }

                # Écrit une ligne JSON en prenant l'objet record
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    except Exception as e:
        print(f"Erreur avec {pdf_path.name}: {e}")
    finally:
        # Ferme le document PDF dans tous les cas
        if 'doc' in locals(): #Locals = dictionnaires des variables dans le contexte courant
            doc.close()

print("Ingestion terminée.")
