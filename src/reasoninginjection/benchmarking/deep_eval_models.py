# wrapper around models so deepeval can use them

from deepeval.models.base_model import DeepEvalBaseLLM, DeepEvalBaseEmbeddingModel
from reasoninginjection.pipeline import Pipeline
from reasoninginjection.core import Conversation, Message, Role, Memory
from reasoninginjection.utils import embed_text

    
class BaselineRAGLLM(DeepEvalBaseLLM):
    def __init__(self, model_name:str, pipeline:Pipeline):
        self.model_name = model_name
        self.pipeline = pipeline

    def load_model(self):
        return self.model_name
    

    def generate(self, prompt: str) -> str:
        conversation = Conversation()
        conversation.add_message(Message(role=Role.USER, content=prompt))
        response_message = self.pipeline.generate_response(conversation)
        return response_message.message_text
        
    
    def generate_with_context(self, prompt: str) -> tuple[str, list[str]]:
        conversation = Conversation()
        conversation.add_message(Message(role=Role.USER, content=prompt))
        response_message = self.pipeline.generate_response(conversation)
        return response_message.message_text, []
        
    
    async def a_generate(self, prompt: str) -> str:
        conversation = Conversation()
        conversation.add_message(Message(role=Role.USER, content=prompt))
        response_message = self.pipeline.generate_response(conversation)
        return response_message.message_text
        
    
    def generate_message(self, conversation: Conversation) -> Message:
        response_message = self.pipeline.generate_response(conversation)
        return response_message
        

    def get_model_name(self):
        return self.model_name
    
class LocalEmbeddingModel(DeepEvalBaseEmbeddingModel):
    def __init__(self):
        self.model_name = "all-MiniLM-L6-v2"

    def load_model(self):
        return self.model_name

    def get_embedding(self, text: str) -> list[float]:
        return embed_text(text).tolist()
    
    def embed_text(self, text: str) -> list[float]:
        return embed_text(text).tolist()
    
    def a_embed_text(self, text: str) -> list[float]:
        return embed_text(text).tolist()
    
    def get_model_name(self):
        return self.model_name
    
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return embed_text(texts).tolist()
    
    def a_embed_texts(self, texts: list[str]) -> list[list[float]]:
        return embed_text(texts).tolist()