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
        conversation = conversation.copy()
        
        
        
        # generate reasoning message with structured format
        reasoning_content = self.generate_reasoning_message(conversation, contexts)
        
        system_instruction = Message(role=Role.SYSTEM, content="You are a helpful assistant. Think about the user's prompt and the relevant facts to form a well-structured response.")
        conversation.messages.insert(0, system_instruction)


        # Use the generator to produce a response
        response_message = self.generator.generate(conversation, context=reasoning_content, **kwargs)

        return response_message
    
    def generate_reasoning_message(self, conversation: Conversation, contexts: list[str]) -> str:
        """
        Generate a structured reasoning message from the provided contexts.

        Parameters
        ----------
        conversation : Conversation
            The conversation to generate the reasoning message for.
        contexts : list of str
            Additional context strings to consider during generation.

        Returns
        -------
        str
            The structured reasoning message.
        """
        
        user_question = conversation.messages[-1]
        
        steps = [
            "Analyze the user's prompt to understand the core question and intent.",
            "Evaluate the provided passages for relevance to the user's question.",
            "Extract and combine relevant information from the passages.",
            "Fix mistakes that were made in previous steps.",
            "Plan the final response structure in a clear and organized manner. Include and repeat all facts needed."
        ]
        
        instructions_content = (
            "You are provided with a user's prompt and several text passages of content. "
            "Your task is to preprocess these passages into a structured format that will help in generating a "
            "clear and concise response. Follow these steps:\n"
            f"# [Step 1] {steps[0]}\n"
            "- Identify the specific question(s) being asked.\n"
            "- Determine the key topics or subjects involved.\n"
            "- Identify any constraints or requirements specified by the user.\n\n"
            f"# [Step 2] {steps[1]}\n"
            "- For each passage, assess its relevance to the identified question(s) and topics.\n"
            "- Determine which passages contain information that directly addresses the user's core intends.\n\n"
            f"# [Step 3] {steps[2]}\n"
            "- From the relevant passages, extract key facts, data points, and insights.\n"
            "- Organize the extracted information by putting related facts together.\n\n"
            f"# [Step 4] {steps[3]}\n"
            "- Identify and correct any mistakes made in the previous steps.\n\n"
            "- Correct any instances where you ignored or contradicted relevant information from the passages.\n"
            f"- If there are no mistakes, state that no corrections are needed.\n\n"
            f"# [Step 5] {steps[4]}\n"
            "- Format as a first-person inner monologue.\n"
            "- First reiterate the user's question(s) and intent to ensure clarity.\n"
            "- Then outline the main points to be covered in the response.\n"
            "- Finally, reiterate the facts and insights one by one in a logical order. Repeat the contents of the passages. Don't refer to the passages itself. The text must include all relevant information without relying on any additional context.\n\n"
            "Format the structured reasoning message as follows:\n"
            f"# [Step 1] {steps[0]}\n"
            "<your analysis here>\n\n"
            f"# [Step 2] {steps[1]}\n"
            "<your evaluation here>\n\n"
            f"# [Step 3] {steps[2]}\n"
            "<your extracted information here>\n\n"
            f"# [Step 4] {steps[3]}\n"
            "<your corrections here>\n\n"
            f"# [Step 5] {steps[4]}\n"
            "<your planned response structure here>\n\n"
            "Now, using the passages provided, create the structured reasoning message."
        )
        
        instructions_message = Message(role=Role.SYSTEM, content=instructions_content)
        
        conversation = Conversation(messages=[instructions_message])
        
        # Here we would implement the preprocessing steps to create the structured reasoning message.
        # For simplicity, we'll just concatenate the contexts with some formatting.
        reasoning_content = f"USER PROMPT: {user_question.content}\n\n"
        
        reasoning_content += "PASSAGES:\n"
        for idx, context in enumerate(contexts):
            reasoning_content += f"[Passage {idx+1}]: {context}\n"
        reasoning_content += "\nNow, please preprocess these passages into a structured reasoning format as per the instructions above."
        
        conversation.add_message(Message(role=Role.USER, content=reasoning_content))
        final_step = None
        max_tries = 3
        tries = 0
        
        instruction_steps = [step for step in steps]
        instruction_steps[-1] += "\nOkay, the user"
        
        while final_step is None and tries < max_tries:
            try:
                tries += 1
                output = self.generator.generate_with_steps(conversation, steps=instruction_steps)
                
                output_steps = self.generator.parse_steps_from_message(output, steps=instruction_steps)
                
                final_step = output_steps[-1]
            except Exception as e:
                print(f"Attempt {tries} to parse steps failed: {e}")
                final_step = None
        
        return final_step or ""
    
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

