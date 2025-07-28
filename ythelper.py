from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings  # Correct import from langchain_community
from langchain_community.vectorstores import FAISS  # Correct import from langchain_community
from langchain_core.prompts import PromptTemplate  # Correct import from langchain_core
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Function to fetch the transcript from YouTube using YouTubeTranscriptApi
def fetch_transcript(video_url: str):
    try:
        # Extract video ID from the URL
        video_id = video_url.split("v=")[-1]

        # Fetch the transcript using YouTubeTranscriptApi
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return transcript
    except Exception as e:
        print(f"Error fetching transcript: {e}")
        return None  # Return None if transcript is not found

# Function to create FAISS database from YouTube video URL
def create_db_from_youtube_video_url(video_url: str) -> FAISS:
    # Fetch the transcript from the video URL
    transcript = fetch_transcript(video_url)
    
    if transcript is None:
        print("Could not load transcript.")
        return None
    
    # Split the transcript into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)
    
    # Create and return the FAISS vector store
    db = FAISS.from_documents(docs, embeddings)
    return db

# Function to get a response based on the query and the FAISS database
def get_response_from_query(db, query, k=4):
    """
    text-davinci-003 can handle up to 4097 tokens. Setting the chunksize to 1000 and k to 4 maximizes
    the number of tokens to analyze.
    """

    if db is None:
        print("No database found.")
        return "Error: No database found."

    # Perform similarity search on the FAISS database
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    # Initialize the LLM model
    llm = OpenAI(model_name="gpt-3.5-turbo-instruct")

    # Define the prompt template
    prompt = PromptTemplate(
        input_variables=["question", "docs"],
        template="""
        You are a helpful assistant that can answer questions about YouTube videos 
        based on the video's transcript.
        
        Answer the following question: {question}
        By searching the following video transcript: {docs}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.
        """,
    )

    # Create a chain to process the prompt and get the response
    chain = LLMChain(llm=llm, prompt=prompt)

    # Run the chain to get the response
    response = chain.run(question=query, docs=docs_page_content)
    response = response.replace("\n", "")  # Clean the response by removing newline characters
    return response
