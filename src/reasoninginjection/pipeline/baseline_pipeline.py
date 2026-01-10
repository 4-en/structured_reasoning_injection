from reasoninginjection.generator import BaseGenerator
from reasoninginjection.core import Conversation, Message, Role

from .pipeline import Pipeline

class BaselinePipeline(Pipeline):
    """
    A baseline pipeline that uses the generator without adding additional context.
    Just returns the response from the underlying LLM.
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
            These will be ignored in the baseline pipeline.

        Returns
        -------
        Message
            The generated response message.
        """


        # Use the generator to produce a response
        response_message = self.generator.generate(conversation, **kwargs)

        return response_message