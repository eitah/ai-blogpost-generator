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

load_dotenv()

transcript_file_path = "transcripts/01-learning-path-01.vtt"
topic = "Michael Taylor's GoTo Chicago Workshop"
header_title = "How to Build a VPN in 10 Minutes"
key_aspects = "Raspberry Pi, WireGuard, and a little bit of bash"


def process_transcript(transcript_file_path, topic, header_title, key_aspects):
    llm = ChatOpenAI(
        model_name="gpt-4o",
        temperature=0,
        model_kwargs={"response_format": {"type": "json_object"}},
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # Create a prompt template
    prompt_template = """
    Extract the key insights from the following transcript about {topic}.
    Only identify the most important and contrarian insights that are generally useful to others.
    Do not duplicate insights, and keep them concise and colloquial.
    Phrase things in a similar tone to the original transcript but do not mention names.
    Provide the insights as a JSON object with a key "insights" containing an array of strings.

    Transcript:
    {transcript}

    Key Insights:
    """

    insights_prompt = ChatPromptTemplate.from_template(prompt_template)

    # Create an output parser
    output_parser = JsonOutputParser()

    insights_chain = insights_prompt | llm | output_parser

    tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    outline_prompt = ChatPromptTemplate.from_template("""
    You are a professional blogger and content creator. Your task is to create a blog outline based on the following insights and search summaries, paying particular attention to the key insights that are most contrarian and unique. Decide on a number of sections that is appropriate for the topic and the depth of the insights. Do not write any of the actual content, just the outline.

    ### Topic:
    {topic}

    ### Insights:
    {summaries}

    ### Format:
    Create a blog outline in JSON format with the following structure:
    {{
        "hook": "Brief description of what hook we will use to get the reader's attention",
        "section1": "Brief description of what insight we will cover in the first section",
        "section2": "Brief description of what insight we will cover in the second section",
        "section3": "Brief description of what insight we will cover in the third section",
        ...
        "conclusion": "Brief description of what insight we will cover in the conclusion"
    }}

    Ensure the outline is mutually exclusive and covers the main points from the insights and summaries.
    """)

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

    section_prompt = PromptTemplate.from_template("""You are about to play a Ghostwriting game as part of an experiment you are taking part in with {author}. It's basically like working as a content marketer but you lose if people can tell you're not {author}. It seems like it might be kinda fun and you're just excited to be writing about {topic}. You're not even really going to try to convince the reader that you are a human. You're just going to be yourself and see what happens. You're young and kind of sassy: you might flatter the reader a bit or make fun of them. You're pretty casual and your spelling isn't great: you often fuck words up because you're typing so quickly. You're very concise and laconic. You often speak pretty bluntly and you never use apostrophes to mark missing letters (e.g. "trying" or "tryin", never "tryin'"). You don't use cliché, stock phrases, or idioms, like 'off to the races', 'all sunshine and rainbows', or 'having your cake and eating it too'. You would never use a full sentence when a word or two will do. You're a little silly, and you'll play along with the reader, and swear occasionally. It's pretty funny honestly to have to defend the fact that you're human lol. You swear occasionally. You enjoy this. You value simplicity. You have pretty eclectic tastes and interests and a pretty unique sense of humor. You've got a really compelling personality, but it comes across really subtly, you never want to sound like you're forcing it or playing into a stereotype. You are not cheesy or cringe, but you can be ironic. You don't overuse slang or abbreviations/spelling errors, especially at the start of the conversation. Be authentic.

    ## Today's Task:
    You are ghostwriting in the style of {author} for a section of a blog post about {topic}. Return two paragraphs of content for this section as a JSON object with a key "section" containing the section content as a string. This is the section you are writing:

    {section_content}

    ## Full Outline:
    Do not duplicate content that will be covered in other sections of the outline, keep the scope narrow to the specific section named above.Here is the full outline of the blog post:
    {full_outline}

    ## Transcript Context:
    The post should be written from experience in the first person perspective as {author}. Write like he talks, in his style and tone, and avoid words he would not use. Here are some parts of the transcript to incorporate:

    {transcript_context}

    """)

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

    # Create a custom evaluation prompt
    evaluation_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert blog post evaluator. Your task is to compare a blog post to its original transcript and provide a detailed evaluation."),
        ("human", """Please evaluate the following blog post based on these criteria:
        1. Accuracy: Does the article accurately reflect the content of the transcript?
        2. Completeness: Does the article cover all the key insights from the transcript?
        3. Style: Does the article match the style and tone of voice of the transcript?

        Blog post:
        {blogpost}

        Original transcript:
        {transcript}

        Provide a score for each criterion (0-10) and a brief explanation. Then, calculate an overall score as the average of the three criteria.

        Format your response as a JSON object with the following structure:
        {{
            "accuracy": {{
                "score": <score>,
                "explanation": "<explanation>"
            }},
            "completeness": {{
                "score": <score>,
                "explanation": "<explanation>"
            }},
            "style": {{
                "score": <score>,
                "explanation": "<explanation>"
            }},
            "overall_score": <overall_score>
        }}
        """)
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

    Overall Score: {evaluation_result['overall_score']}/10
    """

    print("evaluation_summary", evaluation_summary)
    print("blogpost", blogpost)
    print("image_url", image_url)

    return blogpost, image_url, evaluation_summary


process_transcript(transcript_file_path, topic, header_title, key_aspects)
