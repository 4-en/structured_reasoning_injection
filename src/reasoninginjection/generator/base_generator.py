from reasoninginjection.core import Conversation, Message, Role, MessagePart, ThoughtPart
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pydantic import BaseModel
from .config import Config
    
    
@dataclass
class ObjectResult:
    """
    A result object from a generator.
    
    Attributes
    ----------
    success : bool
        Whether the generation was successful.
    result : BaseModel | None
        The generated object, if successful.
    error_message : str | None
        The error message, if not successful.
    """
    
    success: bool
    result: BaseModel | None = None
    reasoning: str | None = None
    message: Message | None = None
    error_message: str | None = None

class BaseGenerator(ABC):
    """
    Base class for a generator.
    A generator generates a response based on the conversation.
    """

    def __init__(self):
        """
        Initialize the generator.
        """
        pass

    @abstractmethod
    def generate(self, conversation:Conversation, context:str=None, response_format:BaseModel=None, **kwargs) -> Message:
        """
        Generate a response based on the conversation.

        Parameters
        ----------
        conversation : Conversation
            The conversation to generate a response for.
        context : str, optional
            The context of the conversation. Can include retrieved information,
            inner monologue, etc.
        response_format : BaseModel, optional
            The format of the model output.
            If None, the output will be a string.
            Otherwise, try to generate json output based on the model.
        kwargs : dict
            Additional keyword arguments for implementation-specific parameters.

        Returns
        -------
        Message
            The response generated.
        """
        pass
    
    @abstractmethod
    def generate_object(self,
                        conversation:Conversation,
                        response_format:BaseModel,
                        context:str=None,
                        allow_reasoning:bool=True,
                        max_attempts:int=3,
                        **kwargs
    ) -> ObjectResult:
        """
        Generate a response based on the conversation and parse it into an object.
        
        Parameters
        ----------
        conversation : Conversation
            The conversation to generate a response for.
        context : str, optional
            The context of the conversation. Can include retrieved information,
            inner monologue, etc.
        response_format : BaseModel, optional
            The format of the model output.
        allow_reasoning : bool, optional
            Whether to allow the model to use reasoning to generate the object.
            If True, the model can generate intermediate reasoning steps.
        max_attempts : int, optional
            The maximum number of attempts to generate a valid object.
            
        Returns
        -------
        ObjectResult
            The result of the object generation.
        """
        raise NotImplementedError()
    
    def generate_with_steps(self, conversation:Conversation, steps:list[str], **kwargs) -> Message:
        """
        Generate a response based on the conversation and intermediate steps.
        
        Guides the response to follow the format of:
        # 1. <step 1><model response>
        # 2. <step 2><model response>
        ...
        # n. <step n><final model response>

        Parameters
        ----------
        conversation : Conversation
            The conversation to generate a response for.
        steps : list of str
            The intermediate reasoning steps to include in the generation.

        Returns
        -------
        Message
            The generated response message.
        """
        
        # by default, use the generate method
        return self.generate(conversation, **kwargs)
    
    def get_config(self) -> Config:
        """
        Get the configuration of the generator.
        
        Returns
        -------
        Config
            The configuration of the generator.
        """
        return Config()
    
    def parse_steps_from_message(self, message:Message, steps:list[str]=None) -> list[str]:
        """
        Parse the intermediate steps from a message.
        
        Parameters
        ----------
        message : Message
            The message to parse the steps from.
        steps : list of str, optional
            The expected steps to parse. If None, parse all steps found.
            If provided, check that the steps match the expected steps.
            
        Returns
        -------
        list of str
            The parsed steps in order, without the headings.
        """
        
        # split steps by using # n. as delimiter
        content = message.message_text
        split_steps = []
        current_step = ""
        lines = content.splitlines()
        for line in lines:
            stripped_line = line.strip()
            if stripped_line.startswith("# [Step "):
                no_hash = stripped_line[8:]  # remove "# [Step "
                one_num_found = False
                while no_hash[0].isdigit():
                    no_hash = no_hash[1:]
                    one_num_found = True
                if one_num_found and no_hash.startswith("]"):
                    # new step found
                    if current_step:
                        stripped_current_step = current_step.strip()
                        if stripped_current_step and stripped_current_step != "":
                            split_steps.append(stripped_current_step)
                    current_step = ""
                    continue
            current_step += line + "\n"
        if current_step:
            split_steps.append(current_step.strip())
        
        
        # if expected steps are provided, check that they match
        if steps is not None:
            if len(split_steps) != len(steps):
                raise ValueError(f"Expected {len(steps)} steps, but found {len(split_steps)} steps.\n\nSteps found:\n" + "\n#############\n".join(split_steps))
        return split_steps
    
    def add_context_as_expert(self, conversation: Conversation, context: str) -> Conversation:
        """
        Add the context as a message from an expert to the conversation.
        
        This is useful for providing additional context to the model when the
        API only supports chat completion like OpenAI, and not direct access
        to token generation.
        
        Example:
        <User message> (prompt)
        <Expert message> (context)
        <Assistant message> (response)

        Parameters
        ----------
        conversation : Conversation
            The conversation to add the context to.
        context : str
            The context to add.

        Returns
        -------
        Conversation
            The conversation with the context added.
        """
        # TODO: test if context before prompt works better
        # (might make it easier for the model to attend to actual question)
        # last_message = conversation.messages[-1]

        summary = f"<Expert> {context}"
        message = Message(summary, Role.ASSISTANT)
        # insert the summary before the actual question
        # otherwise, some models might not be able to attend to the actual question
        # or interpret the summary as the message to respond to
        conversation.messages.append(message)
        return conversation
    
    def add_context_as_reasoning(self, conversation: Conversation, context: str) -> Conversation:
        """
        Add the context as reasoning to the conversation.
        
        This is useful for providing the model more natural access to the context,
        similarly to how it would generate a response based on its reasoning
        process.
        
        Example:
        <User message> (prompt)
        <Assistant message> (context) (response)

        Parameters
        ----------
        conversation : Conversation
            The conversation to add the context to.
        context : str
            The context to add.

        Returns
        -------
        Conversation
            The conversation with the context added.
        """
        reasoning = f"{context}"
        
        reasoning_part = ThoughtPart(reasoning)
        
        message = Message(content=[reasoning_part], role=Role.ASSISTANT)
        conversation.messages.append(message)
        
        self._reasoning_message = message
        return conversation

    def clean_conversation_state(self, conversation: Conversation) -> Conversation:
        """
        Clean the conversation state to remove an injected context
        and avoid duplication.
        The generate method already returns the new Message, so the conversation
        shouldn't include the context message.
        
        Parameters
        ----------
        conversation : Conversation
            The conversation to clean.
        """
        
        # compare last message in conversation to generated message
        if len(conversation.messages) == 0:
            return conversation
        
        # compare last message in conversation to generated message
        if getattr(self, "_reasoning_message", None) is not None:
            if conversation.messages[-1] == self._reasoning_message:
                conversation.messages.pop(-1)
                self._reasoning_message = None