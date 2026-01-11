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
        # Combine contexts into a single string if provided
        if contexts:
            # use LLM to preprocess contexts into structured format (not implemented here)
            expert_content = "\n".join(contexts)
            expert_message = Message(role=Role.EXPERT, content=expert_content)
            conversation.add_message(expert_message)

        # Use the generator to produce a response
        response_message = self.generator.generate(conversation, **kwargs)

        return response_message