import os
import pdfplumber
from markdownify import markdownify
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer
import numpy as np

# 📌 Chemin du dossier contenant les PDFs dans le repo
DATA_DIR = "data"
MD_DIR = "markdown_files"
DB_DIR = "chroma_db"
RESULTS_FILE = "résultats.md"

# 📌 Modèles Hugging Face
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 📌 Modèle d’embedding
GEN_MODEL = "mistralai/Mistral-7B-Instruct"  # 📌 Modèle pour la génération

# 📌 Liste de questions
QUESTIONS = [
    "Qu'est-ce qu'une fonction en Python ?",
    "Comment utiliser une boucle for ?",
    "Quelle est la différence entre while et for ?",
    "Comment gérer les erreurs avec try/except ?",
    "Qu'est-ce qu'une opération CRUD en base de données avec Python ?"
]

# 📌 Chargement du modèle d’embedding
print("🔍 Chargement du modèle d'embedding...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

def embed_texts(texts):
    """Transforme une liste de textes en vecteurs d’embedding"""
    return embedding_model.encode(texts, convert_to_numpy=True)

# 📌 Chargement du modèle Hugging Face pour la génération
print("🚀 Chargement du modèle de génération...")
tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
model = AutoModelForCausalLM.from_pretrained(GEN_MODEL, torch_dtype=torch.float16, device_map="auto")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# 📌 Création des dossiers si nécessaire
os.makedirs(MD_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

# 📌 Conversion PDF → Markdown
def convert_pdf_to_markdown(pdf_path, md_path):
    """Extrait le texte d'un PDF et le convertit en Markdown."""
    md_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                md_text += markdownify(text) + "\n\n"
    with open(md_path, "w", encoding="utf-8") as md_file:
        md_file.write(md_text)

# 📌 Chargement des documents
def load_documents():
    """Charge les fichiers Markdown, en les générant depuis les PDF si nécessaire"""
    docs = []
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(DATA_DIR, filename)
            md_filename = filename.replace(".pdf", ".md")
            md_path = os.path.join(MD_DIR, md_filename)
            if not os.path.exists(md_path):
                print(f"📝 Conversion du PDF en Markdown : {filename}")
                convert_pdf_to_markdown(pdf_path, md_path)
            loader = TextLoader(md_path)
            docs.extend(loader.load())
    return docs

# 📌 Création de la base de vecteurs avec ChromaDB
def create_vectorstore(chunks):
    """Crée et stocke les embeddings dans ChromaDB"""
    print("🔄 Génération des embeddings et stockage dans ChromaDB...")
    texts = [chunk.page_content for chunk in chunks]
    embeddings = embed_texts(texts)  # 📌 Génère les vecteurs d’embedding
    vectorstore = Chroma.from_embeddings(texts, embeddings, persist_directory=DB_DIR)
    return vectorstore

# 📌 Recherche des informations dans Chroma
def retrieve_context(query, k=3, char_limit=1000):
    """Recherche les passages pertinents pour répondre à une question"""
    vectorstore = Chroma(persist_directory=DB_DIR)
    query_embedding = embed_texts([query])  # 📌 Embedding de la question
    results = vectorstore.similarity_search_by_vector(query_embedding[0], k=k)
    context_list = list(dict.fromkeys([res.page_content.strip() for res in results]))
    context = "\n".join(context_list)
    return context if len(context) <= char_limit else context[:char_limit]

# 📌 Génération de réponse
def generate_response(query):
    """Construit le prompt et génère une réponse avec Hugging Face"""
    context = retrieve_context(query)
    if not context:
        return "❌ Aucun contexte pertinent trouvé."
    prompt = f"""
    📚 **Contexte** :
    {context}

    ❓ **Question** : {query}
    """
    response = generator(prompt, max_length=500, temperature=0.7, do_sample=True)
    return response[0]["generated_text"]

# 📌 Exécution automatique des questions
def main():
    print("\n📥 Chargement des documents...")
    documents = load_documents()
    if not documents:
        print("⚠️ Aucun document trouvé dans 'data/'. Ajoutez des PDF et réessayez.")
        return
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100).split_documents(documents)
    create_vectorstore(chunks)
    print("✅ Base vectorielle prête.")

    for question in QUESTIONS:
        response = generate_response(question)
        print(f"\n❓ {question.upper()}\n🤖 {response}\n")

if __name__ == "__main__":
    main()