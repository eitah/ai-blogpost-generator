import fal_client
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tavily import TavilyClient
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI

transcript_file_path = "transcripts/01-learning-path-01.vtt"
prompts_directory = "prompts"
load_dotenv()

topic = "Exploring computer networking with a Raspberry Pi"
header_title = "How to Build a VPN in 10 Minutes"
key_aspects = "Raspberry Pi, WireGuard, and a little bit of bash"


def read_prompt_template(filename):
    with open(os.path.join(prompts_directory, filename), 'r') as f:
        return f.read()


def process_transcript(transcript_file_path, topic, header_title, key_aspects):
    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0,
        model_kwargs={"response_format": {"type": "json_object"}},
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # Create prompt templates from files
    insights_prompt = ChatPromptTemplate.from_template(
        read_prompt_template('insights_prompt.txt'))
    outline_prompt = ChatPromptTemplate.from_template(
        read_prompt_template('outline_prompt.txt'))
    section_prompt = PromptTemplate.from_template(
        read_prompt_template('section_prompt.txt'))

    # Create an output parser
    output_parser = JsonOutputParser()

    insights_chain = insights_prompt | llm | output_parser

    tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    outline_parser = JsonOutputParser()

    outline_chain = outline_prompt | llm | outline_parser

    # Read the transcript file
    with open(transcript_file_path, 'r') as file:
        transcript_content = file.read()

    # Assuming 'transcript_content' variable contains the full transcript text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(transcript_content)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(chunks, embeddings)

    # Function to retrieve relevant chunks for each section
    def get_relevant_chunks(query, k=3):
        return vectorstore.similarity_search(query, k=k)

    section_parser = JsonOutputParser()
    section_chain = section_prompt | llm | section_parser

    def generate_section_content(section, content, full_outline):
        print(f"Generating content for section: {section}")

        relevant_chunks = get_relevant_chunks(section + " " + content, k=5)
        context = "\n\n".join(
            [chunk.page_content for chunk in relevant_chunks])
        return section_chain.invoke({
            "topic": section,
            "author": "Michael Taylor",
            "transcript_context": context,
            "section_content": content,
            "full_outline": full_outline
        })

    llm_mini = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0,
        model_kwargs={"response_format": {"type": "json_object"}},
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # Create a custom evaluation prompt from file
    evaluation_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert blog post evaluator. Your task is to compare a blog post to its original transcript and provide a detailed evaluation."),
        ("human", read_prompt_template('evaluation_prompt_human.txt'))
    ])

    # Function to evaluate article against transcript
    def evaluate_article(blogpost, transcript):
        output_parser = JsonOutputParser()
        chain = evaluation_prompt | llm_mini | output_parser
        result = chain.invoke({
            "blogpost": blogpost,
            "transcript": transcript
        })
        return result

    # Extract insights
    insights = insights_chain.invoke(
        {"topic": topic, "transcript": transcript_content})

    # Create summaries
    summaries = []
    for insight in insights['insights']:
        search_results = tavily.qna_search(
            f"Is this insight contrarian?: {insight}",
            num_results=5
        )
        summaries.append({"insight": insight, "summary": search_results})

    # Create blog outline
    blog_outline = outline_chain.invoke(
        {"topic": topic, "summaries": summaries})

    # Generate blog content
    blog_content = {}
    for section, content in blog_outline.items():
        section_content = generate_section_content(
            section, content, blog_outline)
        blog_content[section] = section_content["section"]

    blogpost = "\n".join(blog_content.values())

    # Evaluate the article
    evaluation_result = evaluate_article(blogpost, transcript_content)

    evaluation_summary = f"""
    Accuracy Score: {evaluation_result['accuracy']['score']}/10
    {evaluation_result['accuracy']['explanation']}

    Completeness Score: {evaluation_result['completeness']['score']}/10
    {evaluation_result['completeness']['explanation']}

    Style Score: {evaluation_result['style']['score']}/10
    {evaluation_result['style']['explanation']}

    Repetitiveness Score: {evaluation_result['repetitiveness']['score']}/10
    {evaluation_result['repetitiveness']['explanation']}

    Overall Score: {evaluation_result['overall_score']}/10
    """

    print("evaluation_summary", evaluation_summary)
    print("blogpost", blogpost)

    return blogpost, evaluation_summary


process_transcript(transcript_file_path, topic, header_title, key_aspects)
