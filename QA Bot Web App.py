import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ibm import WatsonxEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
import gradio as gr

class QA_Bot:

    def __init__(self):
        self.API_KEY = os.environ.get("IBM_CLOUD_API_KEY")
        self.WATSONX_URL = os.environ.get("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
        self.PROJECT_ID = os.environ.get("IBM_PROJECT_ID")
        self.EMBEDDINGS_MODEL = os.environ.get("EMBEDDINGS_MODEL", "ibm/slate-125m-english-rtrvr")

    # IBM WatsonX LLM setup
    def get_llm(self):
        from ibm_watsonx_ai.foundation_models import ModelInference
        from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

        parameters = {
            GenParams.MAX_NEW_TOKENS: 256,
            GenParams.TEMPERATURE: 0.1,
        }

        llm = ModelInference(
            model_id="ibm/granite-13b-instruct-v2",
            params=parameters,
            credentials={
                "apikey": self.API_KEY,
                "url": self.WATSONX_URL,
            },
            project_id=self.PROJECT_ID,
        )
        return llm

    # Step 1: Document Loader
    def document_loader(self, file):
        """Load a PDF file using PyPDFLoader from langchain_community."""
        loader = PyPDFLoader(file)
        loaded_document = loader.load()
        return loaded_document

    # Step 2: Text Splitter
    def text_splitter(self, data):
        """Split loaded documents into manageable text chunks."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=50,
            length_function=len,
        )
        chunks = text_splitter.split_documents(data)
        return chunks


    # Step 3: WatsonX Embeddings
    def watsonx_embedding(self):
        """Generate text embeddings using WatsonxEmbeddings from langchain_ibm."""
        embeddings = WatsonxEmbeddings(
            model_id=self.EMBEDDINGS_MODEL,
            url=self.WATSONX_URL,
            apikey=self.API_KEY,
            project_id=self.PROJECT_ID,
        )
        return embeddings

    # Step 4: Vector Database
    def vector_database(self, chunks):
        """Embed text chunks and store them in a Chroma vector store."""
        embedding_model = self.watsonx_embedding()
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
        )
        return vectordb

    # Step 5: Retriever
    def retriever(self, file):
        """Load, split, embed, and convert documents into a retriever."""
        # Load the PDF document
        loaded_docs = self.document_loader(file)
        # Split into chunks
        chunks = self.text_splitter(loaded_docs)
        # Create vector database
        vectordb = self.vector_database(chunks)
        # Create retriever with similarity search
        retriever_obj = vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3},
        )
        return retriever_obj


    # Step 6: QA Bot with Gradio Interface
    def retriever_qa(self, file, query):
        """Perform question-answering over documents using RAG."""
        llm = self.get_llm()
        retriever_obj = self.retriever(file)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever_obj,
            return_source_documents=False,
        )
        response = qa_chain.invoke({"query": query})
        return response["result"]


    # Gradio Interface
    qa_interface = gr.Interface(
        fn=retriever_qa,
        inputs=[
            gr.File(
                label="Upload PDF Document",
                file_types=[".pdf"],
            ),
            gr.Textbox(
                label="Enter Your Question",
                placeholder="What this paper is talking about?",
                lines=2,
            ),
        ],
        outputs=gr.Textbox(
            label="Answer",
            lines=6,
        ),
        title="QA Bot",
        description=(
            "Upload a PDF document and ask questions about its content. "
            "Powered by IBM WatsonX, LangChain, and ChromaDB."
        ),
    )

if __name__ == "__main__":

    bot = QA_Bot()
    bot.qa_interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
    )