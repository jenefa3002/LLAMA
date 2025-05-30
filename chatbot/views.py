import logging
import os
import torch
from django.http import JsonResponse
from django.views import View
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.conf import settings
from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import time
import threading
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(settings.MODEL_DIR,"llama-2-7b-chat.Q4_K_M.gguf")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

llm_instance = None
db_instance = None
qa_chain = None
embeddings = None
initialization_complete = False
initialization_thread = None

GREETINGS = [
    "Hello! How can I assist you today?",
    "Hi there! What can I help you with?",
    "Greetings! I'm here to help with your questions."
]

APOLOGIES = [
    "I'm sorry, I couldn't find that information.",
    "Apologies, I don't have the answer to that question.",
    "I'm afraid I don't know the answer to that."
]

THANKS_RESPONSES = [
    "You're welcome! Is there anything else I can help with?",
    "Happy to help! Let me know if you have other questions.",
    "Glad I could assist you!"
]

SMALL_TALK = {
    "how are you": "I'm just a computer program, but I'm functioning well! How can I help you?",
    "who are you": "I'm an AI assistant here to help answer your questions.",
    "what can you do": "I can answer questions based on the documents I've been trained on. Try asking me something!"
}

def initialize_components():
    """Initialize components with timeouts and better resource management"""
    global llm_instance, db_instance, qa_chain, embeddings, initialization_complete
    
    try:
        logger.info("Starting component initialization")        
        start = time.time()
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={
                'normalize_embeddings': False, 
                'batch_size': 128
            }
        )
        logger.info(f"Embeddings loaded in {time.time()-start:.2f}s")        
        start = time.time()
        db_instance = FAISS.load_local(
            settings.VECTORSTORE_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )
        logger.info(f"Vector store loaded in {time.time()-start:.2f}s")        
        start = time.time()
        llm_instance = LlamaCpp(
            model_path=MODEL_PATH,
            temperature=0.1,
            max_tokens=256,
            n_ctx=4096,
            n_threads=4,  
            n_gpu_layers=0,  
            n_batch=128,  
            f16_kv=False, 
            callback_manager=CallbackManager([]),# if the callback manager is needed use StreamingStdOutCallbackHandler()
            verbose=False
        )
        logger.info(f"LLM loaded in {time.time()-start:.2f}s")
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm_instance,
            chain_type="stuff",
            retriever=db_instance.as_retriever(
                search_type="similarity", 
                search_kwargs={"k": 1} 
            ),
            return_source_documents=True
        )
        
        try:
            qa_chain.invoke({"query": "test"})
        except Exception as e:
            logger.warning(f"Warm-up failed: {str(e)}")
        
        initialization_complete = True
        logger.info("All components initialized successfully")
        
    except Exception as e:
        logger.error(f"Initialization failed: {str(e)}")
        llm_instance = None
        db_instance = None
        qa_chain = None

logger.info("Starting initialization")
initialize_components()

@method_decorator(csrf_exempt, name='dispatch')
class ChatbotView(View):
    def post(self, request):
        try:
            question = request.POST.get('question', '').strip().lower()
            if not question:
                return JsonResponse({'error': 'Empty question'}, status=400)
            
            if any(greet in question for greet in ["hi", "hello", "hey", "helo", "Olah"]):
                return JsonResponse({'answer': random.choice(GREETINGS), 'sources': []})
                
            if "thank" in question:
                return JsonResponse({'answer': random.choice(THANKS_RESPONSES), 'sources': []})
            
            if "bye" in question or "exit" in question:
                return JsonResponse({'answer': "Goodbye! Have a great day!", 'sources': ["Ari"]})
                
            for phrase, response in SMALL_TALK.items():
                if phrase in question:
                    return JsonResponse({'answer': response, 'sources': []})
            
            if not initialization_complete:
                return JsonResponse({
                    'answer': "System is still initializing. Please try again shortly.",
                    'sources': []
                })
            
            start_time = time.time()
            try:
                result = qa_chain.invoke({"query": question})
                processing_time = time.time() - start_time
                logger.info(f"Processed query in {processing_time:.2f}s")
                
                if not result['result']:
                    return JsonResponse({
                        'answer': random.choice(APOLOGIES),
                        'sources': []
                    })
                
                return JsonResponse({
                    'answer': result['result'][:1000],  
                    'sources': [doc.metadata.get('page', '?') for doc in result['source_documents']]
                })
                
            except Exception as e:
                logger.error(f"Processing failed: {str(e)}")
                return JsonResponse({
                    'answer': "The request timed out or encountered an error.",
                    'sources': []
                })
            
        except Exception as e:
            logger.error(f"Request handling failed: {str(e)}")
            return JsonResponse({
                'answer': "System error. Please try again.",
                'sources': []
            })


class ChatUIView(View):
    def get(self, request):
        return render(request, 'chatbot/index.html')