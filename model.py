import os
import pandas as pd
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

# Load .env only once
load_dotenv(".env")
hugging_face_api_key = os.getenv("HUGGINGFACE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Load Excel and convert to LangChain Documents
excel_data = pd.read_csv("youtube_csv.csv")
excel_documents = [
    Document(page_content=row["input"], metadata={"youtube_url": row["output"]})
    for _, row in excel_data.iterrows()
]

# Load PDF and split into documents
pdf_docs = PyMuPDFLoader("icar_data_updated.pdf").load()
splitter = RecursiveCharacterTextSplitter(
    chunk_size=750, chunk_overlap=150, separators=["\n\n", "\n", ".", " "]
)
split_documents = splitter.split_documents(pdf_docs)

# Create embeddings (loaded once)
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# Vector DBs (loaded once)
pdf_vector_db = FAISS.from_documents(split_documents, embedding)
pdf_retriever = pdf_vector_db.as_retriever()

youtube_vector_db = FAISS.from_documents(excel_documents, embedding)
youtube_retriever = youtube_vector_db.as_retriever()

# Load LLM once
llm = ChatGroq(model="Llama-3.3-70b-Versatile", groq_api_key=groq_api_key)

# Prompt template
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    तुम एक डिजिटल कृषि सहायक हो, जिसे किसानों की सोयाबीन खेती से जुड़ी समस्याओं का समाधान देना है। 
    तुम्हें नीचे दिए गए "संदर्भ" (context) के आधार पर ही उत्तर देना है — कोई भी जानकारी संदर्भ से बाहर नहीं होनी चाहिए।

    उत्तर इस संरचना में होना चाहिए:
    1. एक छोटा और स्पष्ट शीर्षक (Title) लिखो
    2. बुलेट पॉइंट्स में सरल भाषा में समाधान दो
    3. उत्तर पूरी तरह से हिंदी में होना चाहिए ताकि कोई भी किसान आसानी से समझ सके

    अगर संदर्भ में जवाब नहीं है, तो लिखो:  
    "यह जानकारी उपलब्ध दस्तावेज़ में नहीं है।"

    निर्देश:
    1. उत्तर हमेशा हिंदी में दो
    2. उत्तर को एक स्पष्ट शीर्षक के साथ शुरू करो
    3. क्रमांकित बिंदुओं (1, 2, 3...) में उत्तर दो
    4. प्रतीक चिह्नों का प्रयोग मत करो
    5. जवाब केवल संदर्भ से दो
    6. नहीं मिले तो लिखो: "यह जानकारी उपलब्ध दस्तावेज़ में नहीं है।"
    7. शीर्षक में उपयुक्त इमोजी (🌾, 🐛, 💧, 🧪 आदि) लगाओ
    8. संख्यात्मक जानकारी हो तो तालिका में दिखाओ

    संदर्भ:
    {context}

    प्रश्न:
    {question}

    उत्तर (सिर्फ हिंदी में, शीर्षक और बिंदुओं सहित):
    """
)

# QA Chain (created once)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=pdf_retriever,
    chain_type_kwargs={"prompt": prompt_template}
)

# Only this function runs every time (fast)
def final_model(query):
    # PDF Answer
    response_from_pdf_file = qa_chain.run(query)

    # Youtube Links from Excel
    docs = youtube_retriever.get_relevant_documents(query)
    if not docs:
        formatted_links = "कोई लिंक उपलब्ध नहीं है।"
    else:
        youtube_links = docs[0].metadata["youtube_url"]
        split_links = [link.strip() for link in youtube_links.split(",")]
        formatted_links = "\n".join(split_links)

    return f"Question: {query}\n\nAnswer:\n{response_from_pdf_file}\n\nSuggested Youtube Links:\n{formatted_links}"
