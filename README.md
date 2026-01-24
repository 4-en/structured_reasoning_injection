# Structured Reasoning Injection (SRI)

This repository contains a framework for testing and benchmarking different methods of enhancing **context adherence and faithfulness** in Large Language Models (LLMs), particularly within Retrieval-Augmented Generation (RAG) scenarios.

The primary focus of this research is **StructuredReasoningInjection (SRI)**, a method designed to improve how LLMs process retrieved information by injecting a multi-step, structured reasoning process before generating the final output.

---

## Project Overview

In standard RAG, models often struggle with "lost in the middle" phenomena, noise in retrieved passages, or hallucinations when context is dense. This project implements several architectural "pipelines" to compare standard context delivery against structured injection methods.

The goal is to evaluate if forcing the model through an internal "preprocessing" monologue—analyzing intent, evaluating relevance, extracting facts, and self-correcting—leads to higher grounding in the provided source material.

---

## Implemented Pipelines

The project evaluates five distinct strategies for handling context:

### 1. Baseline Pipeline

The control group. It interacts with the LLM directly without any additional context or specific retrieval logic. It serves as a measure of the model's base knowledge and instruction-following capability.

### 2. Expert Pipeline

A standard RAG approach where retrieved passages are labeled as "Expert Context" and injected into the conversation. The model is instructed to treat this context as the absolute truth.

### 3. Structured Expert (SE) Pipeline

An intermediate approach where an LLM first preprocesses the context into an organized summary through a five-step reasoning process. The final summary is then passed to the generator as the primary context.

### 4. Passage Injection (PI) Pipeline

A method that integrates retrieved passages directly into the model's internal reasoning/generation window, adding specific instructions to use the provided passages to answer the prompt. Based on Passage Injection from [Injecting External Knowledge into the Reasoning Process Enhances Retrieval-Augmented Generation](https://arxiv.org/abs/2507.19333).

### 5. Structured Reasoning Injection (SRI) Pipeline

The core experimental method. It forces a multi-stage reasoning chain:

* **Step 1: Analysis** – Understand user intent and constraints.
* **Step 2: Evaluation** – Assess passage relevance.
* **Step 3: Extraction** – Consolidate facts.
* **Step 4: Correction** – Identify and fix errors or contradictions.
* **Step 5: Planning** – Formulate a first-person inner monologue that reiterates facts before the final response.

---

## Technical Architecture

The framework relies on a modular design:

* **`Generator`**: Interfaces with the underlying LLM to handle completions and step-based parsing.
* **`Conversation` & `Message**`: Core objects that maintain state and role-based logic (System, User, Assistant).
* **`Pipeline`**: An abstract base class ensuring all methods implement `generate_response` and `get_description`.

---

## Current Status

Research is ongoing. Initial testing indicates that while SRI provides a highly structured internal state, performance parity with standard Passage Injection is the current benchmark. Most improtantly, SRI can lead to loss of information due to additional processing step (Passages -> Structured Monologue -> Final Response).
Efforts are focused on:

* Refining the self-correction logic (Step 4).
* Optimizing the "Inner Monologue" prompt to reduce verbosity while maintaining fact density.
* Testing across different model architectures to identify if SRI benefits smaller models more than frontier models.
* Improving growing noise introduction by repeated LLM calls.

