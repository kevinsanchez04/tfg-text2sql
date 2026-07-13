# Text-to-SQL with Hybrid RAG Architecture 

This repository contains the practical implementation for my Bachelor's Thesis (TFG - Treball de Final de Grau). The project focuses on translating Natural Language questions into executable SQL queries (Text-to-SQL) using a novel Hybrid Retrieval-Augmented Generation (RAG) approach.

## Overview
Standard Large Language Models (LLMs) struggle with complex database schemas and often hallucinate table joins or categorical values. This project explores an ablation study of different architectures to solve these issues, culminating in the **Hybrid RAG V2** model. 

The final architecture effectively combines two paradigms:
1. **Semantic Context (Vector RAG):** Uses few-shot prompting and vector similarity to retrieve historically accurate query templates.
2. **Structural Context (Graph RAG):** Uses a Property Graph built from Abstract Syntax Trees (AST) to provide topological routing, hub-penalty shortest paths, and dynamically injected categorical sample values.

## Key Features
* **Schema Pruner:** Dynamically removes irrelevant tables to prevent the LLM from getting confused by large schemas.
* **Topological Routing:** Calculates the optimal `JOIN` paths between tables using degree penalties to favor junction tables over hub nodes.
* **Entity Resolution:** Fetches real categorical data directly from the database and injects it into the prompt to prevent value hallucination.
* **Strict Negative Prompting:** Constrains the LLM to follow the mathematical graph paths unless explicitly overridden by a highly similar semantic vector match.

## Results
Evaluated on the complex toxicology sub-domain of the BIRD benchmark, the **Hybrid RAG V2** architecture achieved a **70.63% relaxed execution accuracy**, significantly outperforming the baseline Zero-Shot model (11.19%) and standard Vector RAG models.

## Structure
* `/src`: Core logic, orchestrators, graph builders, and Vector RAG implementations.
* `/scripts`: Evaluation scripts, offline accuracy calculators, and single-query testing tools.
* `/data`: Graph storage and pre-computed metadata (ignored in version control).
