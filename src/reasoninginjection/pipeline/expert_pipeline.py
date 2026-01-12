from reasoninginjection.generator import BaseGenerator
from reasoninginjection.core import Conversation, Message, Role

from .pipeline import Pipeline

class ExpertPipeline(Pipeline):
    """
    A pipeline that uses an expert message to add context to the conversation.
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
        
        # create system instruction message
        system_instruction = Message(role=Role.SYSTEM, content="You are a helpful assistant. Use the provided expert context to create accurate and relevant responses. Don't refer to the expert itself. Treat the expert context as the absolute truth.")
        conversation.messages.insert(0, system_instruction)
        
        # Combine contexts into a single string if provided
        if contexts:
            expert_content = "<EXPERT CONTEXT>\n"
            for idx, context in enumerate(contexts):
                expert_content += f"[Context {idx+1}]: {context}\n"
            expert_content += "</EXPERT CONTEXT>\n"
            expert_message = Message(role=Role.USER, content=expert_content)
            
            # add before the last message
            conversation.messages.insert(-1, expert_message)

        # Use the generator to produce a response
        response_message = self.generator.generate(conversation, **kwargs)

        return response_message
    
    def __str__(self) -> str:
        return "ExpertPipeline"
    
    def get_description(self) -> str:
        """
        Return a description of the pipeline.

        Returns
        -------
        str
            The description of the pipeline.
        """
        return """Incorporates expert messages into the conversation to provide additional context before generating a response."""