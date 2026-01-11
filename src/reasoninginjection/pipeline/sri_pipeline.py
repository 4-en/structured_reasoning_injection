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
    
    def __str__(self) -> str:
        return "StructuredReasoningInjectionPipeline"
    
    def get_description(self) -> str:
        """
        Return a description of the pipeline.

        Returns
        -------
        str
            The description of the pipeline.
        """
        return """Incorporates structured reasoning injection into the reasoning step to provide additional context before generating a response.
The provided passages are preprocessed into a structured format to enhance the reasoning process of the LLM by creating more organized and less overwhelming context."""


"""
Preprocessing steps:
1. Analyze the user's prompt and core intends.
- Identify the specific question(s) being asked.
- Determine the key topics or subjects involved.
- Identify any constraints or requirements specified by the user.

2. Evaluate the provided passages for relevance.
- For each passage, assess its relevance to the identified question(s) and topics.
- Determine which passages contain information that directly addresses the user's core intends.

3. Extract and combine relevant information.
- From the relevant passages, extract key facts, data points, and insights.
- Organize the extracted information by putting related facts together.

4. Plan the final response structure.
- Format as a first-person inner monologue.
- Reiterate the user's question(s) and intent to ensure clarity.
- Outline the main points to be covered in the response.
- Reiterate the facts and insights one by one in a logical order.

Okay, the user...
        
"""