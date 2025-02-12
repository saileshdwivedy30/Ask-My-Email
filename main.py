import streamlit as st
from datetime import datetime, timedelta
import base64
from bs4 import BeautifulSoup
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_core.documents import Document


# Gmail API Authentication
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

# Persistent ChromaDB directory
CHROMA_DB_PATH = "vector_db"


# Authenticate Gmail API
def gmail_authenticate():
    flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
    creds = flow.run_local_server(port=0)
    return build('gmail', 'v1', credentials=creds)


# Store authenticated service in session state to avoid re-authentication
if "gmail_service" not in st.session_state:
    st.session_state.gmail_service = gmail_authenticate()
service = st.session_state.gmail_service  # Use cached Gmail service

# Initialize ChromaDB
def load_or_create_vector_store():
    return Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=OllamaEmbeddings(model="deepseek-r1:14b")
    )


vector_store = load_or_create_vector_store()


# Fetch Promotional Emails from Last N Days
def fetch_promotion_emails(service, days=7, max_results=20):
    date_n_days_ago = (datetime.now() - timedelta(days=days)).strftime('%Y/%m/%d')
    query = f"label:promotions after:{date_n_days_ago}"

    results = service.users().messages().list(userId='me', q=query, maxResults=max_results).execute()
    messages = results.get('messages', [])

    emails = []
    print("\n📩 DEBUG: Fetching Emails...")

    for msg in messages:
        msg_data = service.users().messages().get(userId='me', id=msg['id']).execute()
        payload = msg_data.get('payload', {})
        headers = payload.get('headers', [])

        subject, sender, date = "Unknown", "Unknown", "Unknown"
        for header in headers:
            if header['name'] == 'Subject':
                subject = header['value']
            elif header['name'] == 'From':
                sender = header['value']
            elif header['name'] == 'Date':
                date = header['value']

        # Print extracted fields
        print(f"📧 Extracted: Sender={sender} | Subject={subject} | Date={date}")

        email_body = "No Content found"
        parts = payload.get('parts', [])
        for part in parts:
            if part.get('mimeType') == 'text/html':
                body_data = part['body'].get('data', '')
                try:
                    decoded_data = base64.urlsafe_b64decode(body_data).decode('utf-8')
                    email_body = BeautifulSoup(decoded_data, 'html.parser').get_text()
                except Exception as e:
                    print(f"Error decoding email body: {e}")
                break

        # Print debugging information
        print(f"📧 Sender: {sender} | Subject: {subject} | Date: {date}")

        emails.append(Document(page_content=f"Subject: {subject}\nFrom: {sender}\nDate: {date}\n\n{email_body}"))

    print(f"\n✅ Total Emails Fetched: {len(emails)}\n")
    return emails


# Text Splitting for Indexing
def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    return text_splitter.split_documents(documents)


# Index Promotional Emails Persistently
def index_docs(documents):
    clean_documents = [doc for doc in documents if "From:" in doc.page_content]  # Ensure only valid emails

    print(f"📌 Indexing {len(clean_documents)} valid emails into ChromaDB...")
    for doc in clean_documents:
        sender_info = doc.page_content.split("From: ")[1].split("\n")[
            0] if "From: " in doc.page_content else "Unknown Sender"
        print(f"📥 Indexing email from: {sender_info}")

    vector_store.add_documents(clean_documents)
    print(f"\n✅ Indexed {len(clean_documents)} emails.\n")


# Retrieve Relevant Emails
def retrieve_docs(query):
    results = vector_store.similarity_search(query, k=20)

    print(f"\n🔍 Searching for '{query}'")
    print(f"📂 Retrieved {len(results)} results.\n")

    retrieved_docs = []
    for res in results:
        lines = res.page_content.split("\n")
        sender_info = next((line for line in lines if line.startswith("From:")), "Unknown Sender")

        print(f"📩 Retrieved email from: {sender_info}")
        retrieved_docs.append(res.page_content)

    return results

template = """
You are an intelligent assistant. Your **only job** is to directly answer the user's question 
based on the relevant email content: {context}. 

Here is the question: {question}

Rules:
1. In your answer, always first mention what you understood from the question. 
2. Provide a concise, direct answer to the question.
3. Always refer to the question exactly as asked. 
4. If you can't find an answer in the context, say so.
5. Do not make up stuff that is not provided in the context.
"""


def answer_question(question, documents):
    context = "\n\n".join([doc.page_content for doc in documents])

    # Show what context is being sent to AI
    print("\n🔍 Context sent to AI Model:")
    for i, doc in enumerate(documents, 1):
        print(f"📄 Email {i}: {doc.page_content[:300]}...")  # Show first 300 characters of each doc

    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | OllamaLLM(model="deepseek-r1:14b")

    response = chain.invoke({"question": question, "context": context})

    # Show AI response before displaying
    print(f"\n🤖 AI Response:\n{response}\n")

    return response


# Streamlit UI
st.title("📨 Chat with Your Promotions Emails")

# Fetch New Emails & Index
n_days = st.slider("Select Number of Days", 1, 30, 7)
if st.button("Fetch Emails"):
    with st.spinner("Fetching Promotional Emails..."):
        new_documents = fetch_promotion_emails(service, days=n_days)
        if new_documents:
            chunked_documents = split_text(new_documents)
            index_docs(chunked_documents)
            st.success(f"✅ Indexed {len(new_documents)} new promotional emails!")
        else:
            st.info("No new promotional emails found.")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []  # Store chat history

# Display Previous Chat Messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Handle New User Input
question = st.chat_input("Ask a question about your promotions emails...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})  # Store user input
    st.chat_message("user").write(question)  # Display user input

    # Retrieve and Generate Answer
    related_documents = retrieve_docs(question)
    answer = answer_question(question, related_documents)

    st.session_state.messages.append({"role": "assistant", "content": answer})  # Store assistant response
    st.chat_message("assistant").write(answer)  # Display assistant response
