import os
from django.core.management.base import BaseCommand
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pathlib import Path
import shutil  # Added for cleanup

class Command(BaseCommand):
    help = 'Process PDF document and create vector store'

    def handle(self, *args, **options):
        
        BASE_DIR = Path(__file__).resolve().parent.parent.parent
        PDF_PATH = os.path.join(BASE_DIR, 'chatbot', 'data', 'document.pdf')
        VECTORSTORE_DIR = os.path.join(BASE_DIR, 'chatbot', 'vectorstore')
        if not os.path.exists(PDF_PATH):
            self.stdout.write(self.style.ERROR(f'PDF not found at: {PDF_PATH}'))
            self.stdout.write(self.style.WARNING('Ensure the file exists and path is correct.'))
            return
        try:
            loader = PyPDFLoader(PDF_PATH)
            pages = loader.load_and_split()
            self.stdout.write(self.style.SUCCESS(f'Loaded {len(pages)} pages from PDF'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'PDF loading failed: {str(e)}'))
            return

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False
        )
        texts = text_splitter.split_documents(pages)
        self.stdout.write(self.style.SUCCESS(f'📚 Split into {len(texts)} text chunks'))

        embeddings = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        try:

            if os.path.exists(VECTORSTORE_DIR):
                shutil.rmtree(VECTORSTORE_DIR)
                self.stdout.write(self.style.WARNING('♻️ Deleted old vectorstore'))

            db = FAISS.from_documents(texts, embeddings)
            os.makedirs(VECTORSTORE_DIR, exist_ok=True)
            db.save_local(VECTORSTORE_DIR)
            files_created = os.listdir(VECTORSTORE_DIR)
            if not files_created:
                raise Exception("No files saved in vectorstore directory")
                
            self.stdout.write(self.style.SUCCESS(
                f'Saved vectorstore to: {VECTORSTORE_DIR}\n'
                f'Files created: {", ".join(files_created)}'
            ))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Vectorstore creation failed: {str(e)}'))
            if 'VECTORSTORE_DIR' in locals():
                self.stdout.write(self.style.WARNING(f'Check folder permissions for: {VECTORSTORE_DIR}'))