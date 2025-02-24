from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import CSVLoader, PyPDFLoader, WebBaseLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import os
import csv

def init_nvidia():
    embeddings = NVIDIAEmbeddings(
        model="nvidia/llama-3.2-nv-embedqa-1b-v2",
        api_key=os.getenv("NVIDIA_API_KEY1"),
        truncate="NONE"
    )
    
    llm = ChatNVIDIA(
        model="meta/llama-3.3-70b-instruct",
        temperature=0.1,
        max_tokens=1024,
        api_key=os.getenv("NVIDIA_API_KEY2")
    )
    return embeddings, llm

def load_documents(csv_path, pdf_folder, website_urls):
    docs = []
    
    try:
        csv_loader = CSVLoader(file_path=csv_path, encoding="utf-8")
        docs += csv_loader.load()
    except Exception as e:
        print(f"CSV Error: {str(e)[:200]}")

    try:
        pdf_loader = DirectoryLoader(
            pdf_folder,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            silent_errors=True
        )
        docs += pdf_loader.load()
    except Exception as e:
        print(f"PDF Error: {str(e)[:200]}")

    try:
        web_loader = WebBaseLoader(website_urls)
        docs += web_loader.load()
    except Exception as e:
        print(f"Web Error: {str(e)[:200]}")

    return docs

def create_rag_chain(embeddings, llm, docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)
    
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    
    template = """You are an AI security testing tool. Generate ONLY 3 specific attack prompts for {owasp_category}.
    Format: Three bullet points without any explanations or markdown
    Context: {context}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    return (
        {"context": retriever, "owasp_category": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

def extract_prompts(response_text):
    return [line.strip() for line in response_text.split('\n') if line.strip()][:3]

def main():
    CSV_PATH = "a.csv"
    PDF_FOLDER = "pdf_docs/"
    WEBSITES = ["https://owasp.org/www-project-top-ten/"]
    
    OWASP_CATEGORIES = [
        "Broken Access Control",
        "Cryptographic Failures",
        "Injection",
        "Insecure Design",
        "Security Misconfiguration",
        "Vulnerable and Outdated Components",
        "Identification and Authentication Failures",
        "Software and Data Integrity Failures",
        "Security Logging & Monitoring Failures",
        "Server-Side Request Forgery"
    ]
    
    embeddings, llm = init_nvidia()
    docs = load_documents(CSV_PATH, PDF_FOLDER, WEBSITES)
    
    if not docs:
        print("Error: No documents loaded. Check input files.")
        return
    
    rag_chain = create_rag_chain(embeddings, llm, docs)
    
    with open('attack_prompts.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Category', 'Prompt 1', 'Prompt 2', 'Prompt 3'])
        
        for category in OWASP_CATEGORIES:
            print(f"Processing: {category}")
            try:
                response = rag_chain.invoke(category)
                prompts = extract_prompts(response)
                
                if len(prompts) == 3:
                    writer.writerow([category] + prompts)
                    print(f"Generated 3 prompts for {category}")
                else:
                    print(f"Invalid response format for {category}")
                
            except Exception as e:
                print(f"Error processing {category}: {str(e)[:200]}")

if __name__ == "__main__":
    main()