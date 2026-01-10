# base class for pipelines
from reasoninginjection.generator import BaseGenerator
from reasoninginjection.core import Conversation, Message, Role

from abc import ABC, abstractmethod


class Pipeline(ABC):
    """
    Base class for pipelines that integrate reasoning injection with a generator.
    """

    def __init__(self, generator: BaseGenerator):
        """
        Initialize the pipeline.

        Parameters
        ----------
        generator : BaseGenerator
            The generator to use for generating responses.
        """
        self.generator = generator

    @abstractmethod
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
        pass