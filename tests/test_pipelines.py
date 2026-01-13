import json
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

from reasoninginjection.pipeline import SEPipeline
from reasoninginjection.core import Conversation, Message, Role
from reasoninginjection.generator import LowLevelLlamaCppGenerator



def run_test():
    # Load the JSON
    try:
        with open('test_case.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: 'test_case.json' not found. Please create it first.")
        return

    question = data['question']
    passages = data['passages']
    expected = data['expected_answer']

    print(f"TEST QUESTION: {question}\n")

    # Initialize Pipeline with Debug Generator
    generator = LowLevelLlamaCppGenerator()
    pipeline = SEPipeline(generator)

    # Create Conversation
    conv = Conversation()
    conv.add_message(Message(role=Role.USER, content=question))

    # Run Pipeline
    response = pipeline.generate_response(conversation=conv, contexts=passages)

    print("\n" + "="*40)
    print(" FINAL OUTPUT ")
    print("="*40)
    print(f"ACTUAL RESPONSE:   {response.content}")
    print(f"EXPECTED ANSWER:   {expected}")

if __name__ == "__main__":
    run_test()