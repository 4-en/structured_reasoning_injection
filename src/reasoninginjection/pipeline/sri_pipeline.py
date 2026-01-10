# structured reasoning injection pipeline

from reasoninginjection.generator import BaseGenerator
from reasoninginjection.core import Conversation, Message, Role

from .pipeline import Pipeline

class SRIPipeline(Pipeline):
    """
    A pipeline that integrates reasoning injection with a generator.
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
        context_str = "\n".join(contexts) if contexts else None

        # Use the generator to produce a response
        response_message = self.generator.generate(conversation, context=context_str, **kwargs)

        return response_message