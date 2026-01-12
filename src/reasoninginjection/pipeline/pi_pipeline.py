# passage injection pipeline

from reasoninginjection.generator import BaseGenerator
from reasoninginjection.core import Conversation, Message, Role

from .pipeline import Pipeline

class PIPipeline(Pipeline):
    """
    A pipeline that integrates passage injection with a generator.
    """


    def generate_response(self, conversation: Conversation, contexts: list[str] = None, **kwargs) -> Message:
        """
        Generate a response based on the conversation and optional contexts.

        Parameters
        ----------
        conversation : Conversation
            The conversation to generate a response for.
        contexts : list of str, optional
            Additional context strings to consider during generation.

        Returns
        -------
        Message
            The generated response message.
        """
        
        
        conversation = conversation.copy()
        
        system_instruction = Message(role=Role.SYSTEM, content="You are a helpful assistant. Use the provided passages to create accurate and relevant responses.")
        conversation.messages.insert(0, system_instruction)
        
        context_str = "Passages:\n"
        for idx, context in enumerate(contexts):
            context_str += f"[Passage {idx+1}]: {context}\n"
        context_str += "\nI should use the above passages to help me answer the question.\n"

        # Use the generator to produce a response
        response_message = self.generator.generate(conversation, context=context_str, **kwargs)

        return response_message
    
    def __str__(self) -> str:
        return "PassageInjectionPipeline"
    
    def get_description(self) -> str:
        """
        Return a description of the pipeline.

        Returns
        -------
        str
            The description of the pipeline.
        """
        return """Incorporates passage injection into the conversation to provide additional context before generating a response.
The provided passages are added to the reasoning process of the LLM to enhance its responses."""