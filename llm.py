from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks import get_openai_callback
import os
from dotenv import load_dotenv
import nltk
from nltk.tokenize import sent_tokenize

# Download the punkt tokenizer for sentence splitting
nltk.download('punkt', quiet=True)

# Load environment variables
load_dotenv()

# Set up the Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-pro", 
                             google_api_key=os.getenv("GOOGLE_API_KEY"),
                             temperature=0.7)

# Create a prompt template
prompt = ChatPromptTemplate.from_template(
    "You are a helpful assistant. Provide a concise response to the following: {question}"
)

# Create an LLM chain
chain = LLMChain(llm=llm, prompt=prompt)

def generate_response(question, max_sentences=2):
    try:
        # Generate the response
        with get_openai_callback() as cb:
            response = chain.run(question=question)

        # Split the response into sentences
        sentences = sent_tokenize(response)

        # Limit to max_sentences
        limited_response = ' '.join(sentences[:max_sentences])

        return limited_response

    except Exception as e:
        print(f"An error occurred: {e}")
        return "I apologize, but I encountered an error while processing your request."

# Example usage
if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Assistant: Goodbye!")
            break
        response = generate_response(user_input)
        print("Assistant:", response)