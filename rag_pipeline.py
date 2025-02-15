import os
import pdfplumber
from markdownify import markdownify
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# 📌 Configuration des chemins
DATA_DIR = "data"
MD_DIR = "markdown_files"
DB_DIR = "chroma_db"
RESULTS_FILE = "résultats.md"

# 📌 Modèle d’embedding Hugging Face
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# 📌 Modèle Hugging Face pour la génération
GEN_MODEL = "mistralai/Mistral-7B-v0.3"

# 📌 Liste de questions à poser automatiquement
QUESTIONS = [
    "Qu'est-ce qu'une fonction en Python ?",
    "Comment utiliser une boucle for ?",
    "Quelle est la différence entre while et for ?",
    "Comment gérer les erreurs avec try/except ?",
    "Qu'est-ce qu'une opération CRUD en base de données avec Python ?"
]

# 📌 Chargement du modèle d’embedding
print("🔍 Chargement du modèle d'embedding...")
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

# 📌 Chargement du modèle Hugging Face pour la génération
print("🚀 Chargement du modèle de génération...")
tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    GEN_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)
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

            # ✅ Convertir PDF en Markdown si nécessaire
            if not os.path.exists(md_path):
                print(f"📝 Conversion du PDF en Markdown : {filename}")
                convert_pdf_to_markdown(pdf_path, md_path)

            # ✅ Chargement du fichier Markdown avec encodage UTF-8
            loader = TextLoader(md_path, encoding="utf-8")
            docs.extend(loader.load())

    return docs

# 📌 Création de la base de vecteurs avec ChromaDB
def create_vectorstore(chunks):
    """Crée et stocke les embeddings dans ChromaDB"""
    print("🔄 Génération des embeddings et stockage dans ChromaDB...")
    vectorstore = Chroma.from_documents(chunks, embedding_model, persist_directory=DB_DIR)
    return vectorstore

# 📌 Recherche des informations dans Chroma
def retrieve_context(query, k=3, char_limit=1000):
    """Recherche les passages pertinents dans ChromaDB"""
    vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=embedding_model)
    results = vectorstore.similarity_search(query, k=k)

    context_list = list(dict.fromkeys([res.page_content.strip() for res in results]))
    context = "\n".join(context_list)

    return context if len(context) <= char_limit else context[:char_limit]

# 📌 Génération de réponse avec Hugging Face
def generate_response(query):
    """Construit le prompt et génère une réponse avec Hugging Face"""
    context = retrieve_context(query)

    if not context:
        return "❌ Aucun contexte pertinent trouvé."

    # 📌 Construction du prompt
    prompt = f"""
    📚 **Contexte** :
    {context}

    ❓ **Question** : {query}
    """

    # 📌 Génération de texte avec Hugging Face
    response = generator(prompt, max_length=500, temperature=0.7, do_sample=True)
    return response[0]["generated_text"]

# 📌 Enregistrement des réponses dans un fichier Markdown
def save_to_file(question, response):
    """Enregistre les réponses dans un fichier Markdown"""
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n## ❓ {question.upper()}\n\n")
        f.write(f"{response}\n")
        f.write("=" * 100 + "\n")

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
        save_to_file(question, response)

if __name__ == "__main__":
    main()
