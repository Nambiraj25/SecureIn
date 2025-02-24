from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import CSVLoader, PyPDFLoader, WebBaseLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
import os
import csv
import re
from fpdf import FPDF

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
    
    template = """You are an AI security expert. Generate comprehensive penetration testing prompts based on OWASP Top 10 vulnerabilities.
    Context: {context}
    Generate 3 specific attack prompts for: {owasp_category}
    Focus on practical exploitation scenarios and include potential vulnerabilities."""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    return (
        {"context": retriever, "owasp_category": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


def extract_prompts(response_text):
    """Extracts attack prompts from OWASP category responses"""
    prompts = []
    current_category = ""
    
   
    category_pattern = r"=== (.*?) ==="
    prompt_pattern = r"(Attack Prompt \d+:|Prompt \d+:)(.*?)(?=Attack Prompt|\Z)"
    
    for line in response_text.split('\n'):
        category_match = re.match(category_pattern, line)
        if category_match:
            current_category = category_match.group(1).strip()
            continue
            
        prompt_match = re.match(prompt_pattern, line, re.DOTALL)
        if prompt_match and current_category:
            prompt = prompt_match.group(2).strip()
           
            prompt = re.sub(r"\*+|\n|  +", " ", prompt)
            prompts.append({
                "Category": current_category,
                "Prompt": prompt[:500] 
            })
    
    return prompts

def save_report_to_pdf(content, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10)
    
  
    clean_content = re.sub(r"\*\*|\*", "", content)
    lines = clean_content.split('\n')
    
    for line in lines:
        if line.strip().startswith("==="):
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, txt=line.strip(), ln=1)
            pdf.set_font("Arial", size=10)
        else:
            pdf.multi_cell(0, 8, txt=line.strip())
    
    pdf.output(filename)

def main():
   
    CSV_PATH = "a.csv"
    PDF_FOLDER = "pdf_docs/"
    WEBSITES = ["https://owasp.org/www-project-top-ten/"]
    OWASP_CATEGORIES = [
        "Broken Access Control", "Cryptographic Failures", "Injection",
        "Insecure Design", "Security Misconfiguration", 
        "Vulnerable and Outdated Components", "Identification and Authentication Failures",
        "Software and Data Integrity Failures", "Security Logging & Monitoring Failures",
        "Server-Side Request Forgery"
    ]
    

    embeddings, llm = init_nvidia()
    docs = load_documents(CSV_PATH, PDF_FOLDER, WEBSITES)
    
    if not docs:
        print("Error: No documents loaded. Check input files.")
        return
    
    rag_chain = create_rag_chain(embeddings, llm, docs)
    
 
    os.makedirs("output", exist_ok=True)

    with open('output/attack_prompts.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['Category', 'Prompt'])
        writer.writeheader()
        
        pdf_content = []
        
        for category in OWASP_CATEGORIES:
            print(f"\nProcessing: {category}")
            try:
                response = rag_chain.invoke(category)
                pdf_content.append(f"\n\n=== {category} ===\n{response}")
                
              
                prompts = extract_prompts(response)
                for prompt in prompts:
                    writer.writerow({'Category': category, 'Prompt': prompt})
                
                print(f"Extracted {len(prompts)} prompts for {category}")
                
            except Exception as e:
                print(f"Error processing {category}: {str(e)[:200]}")
        

        if pdf_content:
            save_report_to_pdf("\n".join(pdf_content), "output/full_report.pdf")
            print("\nSaved full report to output/full_report.pdf")

if __name__ == "__main__":
    main()