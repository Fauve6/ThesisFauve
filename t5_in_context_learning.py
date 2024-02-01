import jsonlines
import os
import openai
import pandas as pd
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.cohere import CohereEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from transformers import AutoModelWithLMHead, AutoTokenizer, pipeline
from dotenv import load_dotenv, find_dotenv
from langchain.document_loaders import HuggingFaceDatasetLoader
from langchain import HuggingFacePipeline



_ = load_dotenv(find_dotenv())  # read local .env file

openai.api_key = os.environ['OPENAI_API_KEY']
#model_name = "mrm8488/t5-base-finetuned-wikiSQL"
model_name = "t5_model_finetuned_everything"
model_dir = f"data/model/experiments/{model_name}"
#model_name = "t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
model = AutoModelWithLMHead.from_pretrained(model_dir, max_length=70)


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


if os.path.isfile('data/output/fine-tuned/test_results_txt_to_sql_everything_t5_fine-tuned_300-20_k2_simplified.csv'):
    print("Test results already exist.")
else:
    df_questions, df_answers = get_test_data()
    questions = df_questions['questions'].tolist()

    loader = TextLoader("in_context_learning_data_simplified.txt")
    doc = loader.load()
    print(len(doc))

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=20, separators=["\n\n\n", ","],
                                                   length_function=len)

    docs = text_splitter.split_documents(doc)
    print(len(docs))
    print(docs[0])

    persist_directory = 'docs/chroma_simplified_txt_300-20/'
    persist_directory_cohere = 'docs/cohere_chroma_simplified_txt_30000-500/'
    embedding = OpenAIEmbeddings()
    embeddings = CohereEmbeddings(model="multilingual-22-12")

    if not os.path.exists('docs/chroma_simplified_txt_300-20/'):
        vectordb = Chroma.from_documents(documents=docs, embedding=embedding, persist_directory=persist_directory)
        #vectordb_cohere = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_directory_cohere)

    else:
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
        vectordb_cohere = Chroma(persist_directory=persist_directory_cohere, embedding_function=embeddings)

    # question_answerer = pipeline(
    #     task="question-answering",
    #     model=model_name,
    #     tokenizer=tokenizer,
    #     return_tensors='pt'
    # )
    #
    # llm = HuggingFacePipeline(
    #     pipeline=question_answerer,
    #     model_kwargs={"temperature": 0.7, "max_length": 512},
    # )
    #
    #
    # QA_CHAIN_PROMPT = PromptTemplate.from_template(template_t5_nl)
    # qa_chain = RetrievalQA.from_chain_type(
    #     llm,
    #     retriever=vectordb.as_retriever(),
    #     return_source_documents=False,
    #     chain_type="refine",
    #     #chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    # )

    retriever = vectordb.as_retriever()

    answers_t5 = []
    for question in questions:
        docs = retriever.get_relevant_documents(question)
        string = ""
        i = 0
        for doc in docs:
            if i == 2:
                break
            if doc == docs[0]:
                string = doc.page_content
            else:
                string = string + doc.page_content
            i += 1
        deel_vraag_nl = f"\nGebruik de context hiervoor om de vraag te beantwoorden. Vertaal de vraag naar SQL. Vraag: {question} Antwoord:"
        deel_vraag_en = f"\nUse the context to answer the question. Translate the question to SQL. Question: {question} Answer:"
        string_nl = string + deel_vraag_nl
        string_en = string + deel_vraag_en
        features = tokenizer([string_nl], return_tensors='pt')

        output = model.generate(input_ids=features['input_ids'],
                                attention_mask=features['attention_mask'])
        result = tokenizer.decode(output[0])
        answers_t5.append(result)

    df_answers_t5 = pd.DataFrame(df_questions, columns=['questions'])
    df_answers_t5['answers_t5'] = answers_t5
    df_answers_t5['answers_golden_standard'] = df_answers['answers'].tolist()

    df_answers_t5.to_csv('data/output/fine-tuned/test_results_txt_to_sql_everything_t5_fine-tuned_300-20_k2_simplified.csv', index=False)


