import logging
from typing import Optional
import os
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from .config import RAGConfig
from .exceptions import GenerationError

logger = logging.getLogger(__name__)

class ResponseGenerator:
    """Handles response generation using LLM"""
    
    def __init__(self, config: RAGConfig, retriever):
        self.config = config
        self.llm = None
        self.qa_chain = None
        self._initialize_llm()
        self._initialize_qa_chain(retriever)
    
    def _initialize_llm(self) -> None:
        """Initialize LLM (OpenAI or OpenRouter via OpenAI-compatible API)"""
        try:
            # Prefer explicit keys; fall back to env set by config when using OpenRouter
            api_key = (
                self.config.openai_api_key
                or getattr(self.config, "", None)
                or os.getenv("OPENAI_API_KEY")
            )

            # Respect base URL if set (e.g., OpenRouter: https://openrouter.ai/api/v1)
            base_url = os.getenv("OPENAI_API_BASE")

            init_kwargs = {
                "model_name": self.config.model_name,
                "temperature": self.config.temperature,
            }
            if api_key:
                init_kwargs["openai_api_key"] = api_key
            if base_url:
                # Newer langchain-openai uses `base_url`; older may use `openai_api_base`
                init_kwargs["base_url"] = base_url

            self.llm = ChatOpenAI(**init_kwargs)
            logger.info("✅ LLM initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise GenerationError(f"LLM initialization failed: {e}")
    
    def _initialize_qa_chain(self, retriever) -> None:
        """Initialize QA chain with custom prompt"""
        try:
            template = """Use the following pieces of context to answer the question at the end. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            
            Context from your knowledge base:
            {context}
            
            Question: {question}
            Helpful Answer:"""
            
            qa_chain_prompt = PromptTemplate.from_template(template)
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": qa_chain_prompt}
            )
            
            logger.info("✅ QA chain initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize QA chain: {e}")
            raise GenerationError(f"QA chain initialization failed: {e}")
    
    def generate(self, query: str) -> dict:
        """Generate response using QA chain"""
        try:
            if not self.qa_chain:
                raise GenerationError("QA chain not initialized")
            
            result = self.qa_chain.invoke({"query": query})
            return result
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            raise GenerationError(f"Failed to generate response: {e}")