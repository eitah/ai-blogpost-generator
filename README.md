# AI Blog Post Generator

## Overview

This project is a proof of concept for an AI blog post generator that:

1. Takes a transcript as input
2. Evaluates the transcript for key insights and information
3. Generates a blog post based on those insights
4. Evaluates the generated blog post against the original transcript

The goal is to create blog posts that accurately capture and convey the key information from source material while maintaining readability and engagement.

## Current Status

This is an early proof of concept focused on the core workflow. The current implementation:

- Takes a single transcript file as input
- Uses GPT-4 for both analysis and generation
- Outputs a blog post and evaluation metrics
- Provides basic scoring on accuracy, completeness, and style

## Limitations

As a POC, there are several known limitations:

- Only processes single transcript inputs
- No tuning of prompts or parameters
- Basic evaluation metrics
- No style customization
- No content formatting or structure options

See [next-steps.md](next-steps.md) for planned improvements and future direction.

## Original Readme continues below

### GoTo Chicago Workshop: AI Blog Post Generator

There are an emerging set of components that have proven useful in solving the common problems you run into when building applications powered by generative AI. We discuss the generative AI stack and explain from first principles why each component exists, and when they are appropriate to use. You'll build a working prototype of an AI product by chaining multiple components together using LangChain.

The application we'll build is an AI blog post generator, which researches a given topic and summarizes the search results, before generating a comprehensive post based on the research, written in your writing style rather than sounding like AI. We’ll cover advanced topics in applied AI, such as multimodal vision and image generation, open-source models like LLaMA 3, RAG (Retrieval Augmented Generation), as well as the basics like building a simple user interface for AI applications.

Basic experience with Python programming and access to an OpenAI developer account with a payment method is required to follow along, as well as a free Replit account to avoid any dev environment setup issues. You'll get access to all of the code to use in your own applications.

### Setup

Note: We must use .env here not direnv because of the python venv.

1. Create a new virtual environment:

   ```bash
   python3 -m venv venv
   ```

2. Activate the virtual environment:
   - On Windows:

     ```bash
     venv\Scripts\activate
     ```

   - On macOS and Linux:

     ```bash
     source venv/bin/activate
     ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```
