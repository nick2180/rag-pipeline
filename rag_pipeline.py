import os
import pdfplumber
from markdownify import markdownify
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import ollama

# 🛠 Configuration
DATA_DIR = "data"  # Dossier contenant les PDF
MD_DIR = "markdown_files"  # Dossier où stocker les fichiers Markdown
DB_DIR = "chroma_db"  # Base de données ChromaDB
EMBED_MODEL = "nomic-embed-text"  # Modèle d'embedding utilisé par Ollama
GEN_MODEL = "mistral"  # Modèle pour la génération de réponses
RESULTS_FILE = "résultats.md"  # Fichier où enregistrer les résultats

# Liste de questions prédéfinies 📌
QUESTIONS = [
    "Qu'est-ce qu'une fonction en Python ?",
    "Comment utiliser une boucle for en Python ?",
    "Qu'est-ce qu'une condition if-else en Python ?",
    "Comment utiliser le module random en Python ?",
    "Qu'est-ce qu'une opération CRUD en base de données avec Python ?",
    "Comment utiliser une liste en Python ?",
	"Quelle est la différence entre while et for ?",
	"Comment gérer les erreurs avec try/except ?"
]

# Création des dossiers si nécessaire
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MD_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# 📌 CONVERSION PDF -> MARKDOWN
def convert_pdf_to_markdown(pdf_path, md_path):
    """Extrait le texte d'un PDF et le convertit en Markdown."""
    md_text = ""
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                md_text += markdownify(text) + "\n\n"  # Convertir en Markdown
    
    with open(md_path, "w", encoding="utf-8") as md_file:
        md_file.write(md_text)

# 📌 CHARGEMENT DES DOCUMENTS
def load_documents(data_dir, md_dir):
    """Charge les fichiers Markdown, en les générant depuis les PDF si nécessaire"""
    docs = []

    for filename in os.listdir(data_dir):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, filename)
            md_filename = filename.replace(".pdf", ".md")
            md_path = os.path.join(md_dir, md_filename)

            # Convertir le PDF en Markdown s'il n'existe pas encore
            if not os.path.exists(md_path):
                print(f"📝 Conversion du PDF en Markdown : {filename}")
                convert_pdf_to_markdown(pdf_path, md_path)

            # Charger le fichier Markdown
            loader = TextLoader(md_path)
            docs.extend(loader.load())

    return docs

# 📌 TRAITEMENT : DÉCOUPE EN CHUNKS
def process_documents(docs):
    """Divise les documents en morceaux pour l'indexation"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return text_splitter.split_documents(docs)

# 📌 STOCKAGE DES EMBEDDINGS DANS CHROMA
def create_vectorstore(chunks):
    """Crée et stocke les embeddings dans ChromaDB"""
    print("🔄 Génération des embeddings et stockage dans ChromaDB...")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=DB_DIR)
    return vectorstore

# 📌 RECHERCHE DES INFORMATIONS DANS CHROMA
def retrieve_context(query, min_k=2, max_k=5, char_limit=1000):
    """Recherche les passages pertinents en ajustant dynamiquement k"""
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    for k in range(min_k, max_k+1):  # Essai avec différents k
        results = vectorstore.max_marginal_relevance_search(query, k=k, fetch_k=10, lambda_mult=0.5)
        
        # Supprime les doublons tout en conservant l'ordre
        context_list = list(dict.fromkeys([res.page_content.strip() for res in results]))
        context = "\n".join(context_list)

        if len(context) > char_limit:
            break  # Arrête la récupération si trop de texte est ajouté
    
    print("\n📜 Contexte utilisé pour la réponse :\n", context)  # Débogage
    return context

# 📌 GÉNÉRATION DE RÉPONSE AVEC OLLAMA
def generate_response(query):
    """Construit le prompt et génère la réponse avec Ollama"""
    context = retrieve_context(query)

    if not context:
        return "❌ Aucun contexte pertinent trouvé. Vérifiez vos cours."

    prompt = f"""
Tu es un assistant qui **doit uniquement retranscrire le contenu des cours**.

🚨 **Règles strictes** :
1. **Ne jamais inventer** ou compléter les informations.
2. **Ne répondre qu’avec des extraits exacts** du cours.
3. Si l'information demandée **n'est pas dans le contexte**, répondre :
   **"Je ne sais pas. Vérifiez vos cours."**

📚 **Extrait du cours** :
{context}

❓ **Question de l'élève** : {query}

💡 **Réponds en formatant clairement la réponse :**
- **Titre de la section concernée**
- **Extrait exact du cours**
- **Exemple de code (si applicable)**
- **Donne la sortie du code (si applicable)**
"""
    response = ollama.chat(model=GEN_MODEL, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

# 📌 ENREGISTREMENT DES RÉPONSES DANS UN FICHIER MARKDOWN
def save_to_file(question, response):
    """Enregistre les réponses dans un fichier Markdown"""
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n## ❓ {question.upper()}\n\n")
        f.write(f"{response}\n")
        f.write("=" * 100 + "\n")

# 📌 EXÉCUTION DU PIPELINE
def main():
    """Exécute toutes les étapes du RAG"""
    print("\n📥 Chargement des documents...")
    documents = load_documents(DATA_DIR, MD_DIR)

    if not documents:
        print("⚠️ Aucun document trouvé dans 'data/'. Ajoutez des PDF et réessayez.")
        return

    print(f"📃 {len(documents)} documents trouvés. Découpage en chunks...")
    chunks = process_documents(documents)

    print(f"🔍 {len(chunks)} chunks générés. Création de la base de vecteurs...")
    create_vectorstore(chunks)
    print("✅ Base de données vectorielle créée et sauvegardée.")

    print("\n💡 **DÉBUT DES QUESTIONS AUTOMATIQUES** 💡\n")
    for question in QUESTIONS:
        print("=" * 100)
        print(f"\n❓ **QUESTION** : {question.upper()}\n")
        response = generate_response(question)
        print(f"\n🤖 **RÉPONSE D’OLLAMA** :\n{response}\n")
        save_to_file(question, response)

if __name__ == "__main__":
    main()