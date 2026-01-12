from reasoninginjection.generator import BaseGenerator
from reasoninginjection.core import Conversation, Message, Role

from .pipeline import Pipeline

class SEPipeline(Pipeline):
    """
    A pipeline that uses a structured expert message to add context to the conversation.
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
        
        
        
        # generate expert message with structured format
        expert_content = self.generate_expert_message(conversation, contexts)
        
        system_instruction = Message(role=Role.SYSTEM, content="You are a helpful assistant. Use the provided passages to create accurate and relevant responses.")
        conversation.messages.insert(0, system_instruction)
        
        expert_message = Message(role=Role.USER, content="<EXPERT CONTEXT>\n" + expert_content + "\n</EXPERT CONTEXT>\n")
        
        # add before the last message
        conversation.messages.insert(-1, expert_message)

        # Use the generator to produce a response
        response_message = self.generator.generate(conversation, **kwargs)

        return response_message
    
    def generate_expert_message(self, conversation: Conversation, contexts: list[str]) -> str:
        """
        Generate a structured expert message from the provided contexts.

        Parameters
        ----------
        conversation : Conversation
            The conversation to generate the expert message for.
        contexts : list of str
            Additional context strings to consider during generation.

        Returns
        -------
        str
            The structured expert message content.
        """
        expert_content = ""
        for idx, context in enumerate(contexts):
            expert_content += f"[Passage {idx+1}]: {context}\n"
        return expert_content
    
    def __str__(self) -> str:
        return "StructuredExpertPipeline"
    
    def get_description(self) -> str:
        """
        Return a description of the pipeline.

        Returns
        -------
        str
            The description of the pipeline.
        """
        return """Incorporates structured expert messages into the conversation to provide additional context before generating a response.
The provided passages are preprocessed into a structured format to enhance the reasoning process of the LLM by creating more organized and less overwhelming context."""