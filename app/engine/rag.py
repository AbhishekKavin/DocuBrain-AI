import logging
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse
import json

logger = logging.getLogger(__name__)
load_dotenv()

class RAGEngine:
    """
    A RAG (Retrieval-Augmented Generation) engine that retrieves relevant 
    document chunks and generates answers using GPT-4.
    """
    def __init__(self, index_path: str = "faiss_index"):
        self.embeddings = OpenAIEmbeddings(model= "text-embedding-3-small")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # Load the index from disk
        try:
            self.vector_store = FAISS.load_local(
                index_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            self.retriever = self.vector_store.as_retriever(
                search_type = "mmr",
                search_kwargs={"k":5, "fetch_k":10}
                )
            logger.info(f"Successfully loaded FAISS index from disk for retreival.")

        except Exception as e:
            logger.error(f"Error loading FAISS index from disk: {e}")
            raise

    async def stream_query(self, question: str):
        """
        Retrieve relevant documents and generate an answer from the context.
    
        Returns:
            tuple: (answer, sources) where sources are unique source filenames.
        """
        try:
            # Extract Name  (Identity check)
            entity_extraction_prompt = f"Identify the person being asked about in: '{question}'. Return ONLY the full name. If it's a general question about multiple people or no specific person, return 'None'."
            target_response = await self.llm.ainvoke(entity_extraction_prompt)
            target_name = target_response.content.strip()

            # Retrieve relevant documents to get sources
            docs = self.retriever.invoke(question)

            if target_name != "None":
                logger.info(f"Targetting specific entity: {target_name} ")
                final_docs = [
                    doc for doc in docs
                    if target_name.lower() in doc.page_content.lower()
                    or target_name.lower() in doc.metadata.get("source","").lower()
                ]

                if not final_docs:
                    final_docs = docs
            else:
                logger.info("No specific entity identified, using all retrieved documents.")
                final_docs = docs
            # Extract unique sources from the retrieved documents
            sources = list(set([doc.metadata.get("source","Unknown") for doc in final_docs]))
            # Build context string for the LLM
            context = "\n\n".join([doc.page_content for doc in final_docs])

            yield f"SOURCES: {json.dumps(sources)}\n\n"

            # Defining the prompt template
            template = """You are a Career Intelligence Bot. 
            If a specific person is mentioned, answer only about them. 
            If no person is mentioned, provide a collective summary or answer based on all context provided.

            Context:
            {context}

            Question: {question}
            """

            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | self.llm | StrOutputParser()

            async for chunk in chain.astream({"context": context, "question": question}):
                yield chunk
        
        except Exception as e:
            logger.error(f"Error in RAG query logic: {e}", exc_info=True)
            raise

if __name__ == "__main__":
    rag = RAGEngine()
    chain = rag.get_chain()
    response = chain.invoke("Summarize the key experience mentioned in these documents.")
    print(f"\nAI Response:\n{response}")

