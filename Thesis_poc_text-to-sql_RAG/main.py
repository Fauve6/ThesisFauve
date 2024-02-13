import os
import openai
import jsonlines
import time
import pandas as pd
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings.cohere import CohereEmbeddings


_ = load_dotenv(find_dotenv())  # read local .env file

openai.api_key = os.environ['OPENAI_API_KEY']

def get_test_data():
    question_line = []
    answer_line = []
    with jsonlines.open('data/test/test_everything.jsonl') as f:
        for line in f.iter():
            question_line.append(line['messages'][1]['content'])
            answer_line.append(line['messages'][2]['content'])
    questions_df = pd.DataFrame(question_line, columns=["questions"])
    answers_df = pd.DataFrame(answer_line, columns=["answers"])

    return questions_df, answers_df


if not os.path.isfile('data/output/fine-tuned/test_results_txt_to_sql_everything_1106_fine-tuned_english_schema_test_multi_20000-300_k2_simplified.csv'):
    print("Test results already exist.")
else:
    df_questions, df_answers = get_test_data()
    questions = df_questions['questions'].tolist()

    loader = TextLoader("in_context_learning_data_simplified_english.txt")
    doc = loader.load()
    print(len(doc))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=20000, chunk_overlap=300, separators=["\n\n\n", ","],
                                                   length_function=len)

    docs = text_splitter.split_documents(doc)
    print(len(docs))
    print(docs[0])

    persist_directory = 'docs/chroma_txt_simplified_20000-300/'
    persist_directory_cohere = 'docs/cohere_chroma_simplified_txt_30000-500/'
    embedding = OpenAIEmbeddings()
    embeddings = CohereEmbeddings(model="multilingual-22-12")

    if os.path.exists('docs/chroma/'):
        #embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb = Chroma.from_documents(documents=docs, embedding=embedding, persist_directory=persist_directory)
        #vectordb_cohere = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_directory_cohere)
        #vectordb = Qdrant.from_documents(docs, embeddings, location=persist_directory, collection_name="my_documents", distance_func="Dot")

    else:
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
        #vectordb_cohere = Chroma(persist_directory=persist_directory_cohere, embedding_function=embeddings)
    #print(vectordb._collection.count())

    #llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106", temperature=0)
    # llm = Ollama(model="codellama",
    #              callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))


    template = """Gebruik de context om de vraag aan het einde te beantwoorden. 
    Geef als antwoord een SQL-query. Deze SQL-query moet worden kunnen uitgevoerd op een database. 
    Deze database heeft als schema de meegegeven context. Houd er rekening mee dat tabel en kolom namen soms in het Engels zijn.
    Voor het beantwoorden van de vraag moet je eerst kijken in de context welke tabel- en kolomnamen er zijn die relevant 
    zijn voor het beantwoorden van de vraag voordat je verder gaat met het bedenken van een antwoord.
    Als je het antwoord niet weet, zeg dan gewoon dat je het niet weet, probeer geen antwoord te verzinnen.
    {context}
    Vraag: {question}
    Antwoord:"""

    template2 = """Gebruik de context om de vraag aan het einde te beantwoorden. 
    Geef als antwoord een SQL-query. Deze SQL-query moet worden kunnen uitgevoerd op een database met als schema de meegegeven context. 
    Houd er rekening mee dat tabel- en kolomnamen soms in het Engels zijn. In de context staan de tabel- en kolomnamen die in de database voorkomen. 
    Gebruik alleen namen in de SQL query die in de context staan. Achter alle kolomnamen (die tussen '' staan) staat welk type data in die kolom staat. 
    Als er enum() staat, dan staat tussen () welke opties er in die kolom voorkomen. Let hier op bij het maken van de SQL code.
    Voor het beantwoorden van de vraag moet je eerst kijken in de context welke tabel- en kolomnamen er zijn die relevant 
    zijn voor het beantwoorden van de vraag voordat je verder gaat met het bedenken van SQL query.
    Als je het antwoord niet weet, zeg dan gewoon dat je het niet weet, probeer geen antwoord te verzinnen.
    {context}
    Vraag: {question}
    Antwoord:"""

    templateNL = """Gebruik de context om de vraag aan het einde te beantwoorden. 
        Geef als antwoord een SQL-query. Deze SQL-query moet worden kunnen uitgevoerd op een database met als schema de meegegeven context. 
        In de context staan de tabel- en kolomnamen die in de database voorkomen. 
        Gebruik alleen namen in de SQL query die in de context staan. Achter alle kolomnamen (die tussen '' staan) staat welk type data in die kolom staat. 
        Als er enum() staat, dan staat tussen () welke opties er in die kolom voorkomen. Let hier op bij het maken van de SQL code.
        Voor het beantwoorden van de vraag moet je eerst kijken in de context welke tabel- en kolomnamen er zijn die relevant 
        zijn voor het beantwoorden van de vraag voordat je verder gaat met het bedenken van SQL query.
        Als je het antwoord niet weet, zeg dan gewoon dat je het niet weet, probeer geen antwoord te verzinnen.
        {context}
        Vraag: {question}
        Antwoord:"""

    templateEN = """Use context to answer the question at the end. 
        Provide an SQL query as the answer. This SQL query should be able to be executed on a database with the provided context as the schema. 
        The context contains the table and column names that appear in the database. 
        Use only names in the SQL query that are in the context. After the column name (enclosed in '') it is listed what type of data is in that column. 
        If it says enum(), then between () it says what options appear in that column. Pay attention to this when creating the SQL code.
        All table and column names should be in English.
        Before answering the question, look in the context to see what table and column names are relevant 
        to answering the question before you proceed to devise SQL query.
        If you don't know the answer, just say you don't know, don't try to come up with an answer.
        {context}
        Question: {question}
        Answer:"""

    templateNL_testEN = """Gebruik de context om de vraag aan het einde te beantwoorden. 
            Geef als antwoord een SQL-query. Deze SQL-query moet worden kunnen uitgevoerd op een database met als schema de meegegeven context. 
            In de context staan de tabel- en kolomnamen die in de database voorkomen. De Engelse vertaling van de tabel- en kolomnamen kan ook voorkomen.
            Gebruik alleen namen in de SQL query die in de context staan of hun vertaling. Achter alle kolomnamen (die tussen '' staan) staat welk type data in die kolom staat. 
            Als er enum() staat, dan staat tussen () welke opties er in die kolom voorkomen. Let hier op bij het maken van de SQL code.
            Voor het beantwoorden van de vraag moet je eerst kijken in de context welke tabel- en kolomnamen er zijn die relevant 
            zijn voor het beantwoorden van de vraag voordat je verder gaat met het bedenken van SQL query.
            Als je het antwoord niet weet, zeg dan gewoon dat je het niet weet, probeer geen antwoord te verzinnen.
            {context}
            Vraag: {question}
            Antwoord:"""

    templateEN_testNL = """Use context to answer the question at the end. 
            Provide an SQL query as the answer. This SQL query should be able to be executed on a database with the provided context as the schema. 
            The context contains the table and column names that appear in the database. The Dutch translation of table and column names may also occur.
            Use only names in the SQL query that are in the context or their translation. After the column name (enclosed in '') it is listed what type of data is in that column. 
            If it says enum(), then between () it says what options appear in that column. Pay attention to this when creating the SQL code.
            All table and column names should be in English.
            Before answering the question, look in the context to see what table and column names are relevant 
            to answering the question before you proceed to devise SQL query.
            If you don't know the answer, just say you don't know, don't try to come up with an answer.
            {context}
            Question: {question}
            Answer:"""

    template_llama = """Gebruik de context om de vraag te beantwoorden. Geef als antwoord een SELECT SQL statement die kan 
    worden uitgevoerd op de database met de context als schema. 
    Gebruik de context alleen voor het opzoeken van tabel- en kolomnamen.
    {context}
    Question: {question}
    Helpful Answer:"""

    QA_CHAIN_PROMPT = PromptTemplate.from_template(templateEN_testNL )
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(search_kwargs={'k': 2}),
        #retriever=vectordb.as_retriever(search_kwargs={'k': 3}),
        return_source_documents=True,
        #verbose=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    answers_GPT3_5 = []
    result, tmp = [], []
    for value in questions:
        if len(tmp) == 7:
            result.append(tmp)
            tmp = []
        tmp.append(value)
    if tmp:  # add last bucket
        result.append(tmp)

    for questions_list in result:
        for question in questions_list:
            result = qa_chain({"query": question})
            print(result["source_documents"])
            answers_GPT3_5.append(result["result"])
        #time.sleep(60)
    df_answers_GPT = pd.DataFrame(df_questions, columns=['questions'])
    df_answers_GPT['answers_GPT3_5'] = answers_GPT3_5
    df_answers_GPT['answers_golden_standard'] = df_answers['answers'].tolist()

    df_answers_GPT.to_csv('data/output/fine-tuned/test_results_txt_to_sql_everything_1106_fine-tuned_english_schema_test_multi_20000-300_k2_simplified.csv', index=False)


