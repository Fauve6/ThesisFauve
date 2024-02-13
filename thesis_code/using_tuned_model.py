import os
import openai
import jsonlines
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]


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


if os.path.isfile('data/output/test_results_txt_to_sql_train_english_0613_test_multi.csv'):
    print("Test results already exist.")
else:
    df_questions, df_answers = get_test_data()
    questions = df_questions['questions'].tolist()

    answers_GPT3_5 = []
    for question in questions:
        completion = openai.ChatCompletion.create(
            model="ft:gpt-3.5-turbo-0613:personal:english-0613:8mfqZ9ch",
            messages=[
                {"role": "system", "content": "Bokito is een chatbot die natuurlijke taal omzet naar SQL code"},
                {"role": "user", "content": question}
            ]
        )
        answers_GPT3_5.append(completion.choices[0].message['content'])

    df_answers_GPT = pd.DataFrame(df_questions, columns=['questions'])
    df_answers_GPT['answers_GPT3_5'] = answers_GPT3_5
    df_answers_GPT['answers_golden_standard'] = df_answers['answers'].tolist()

    df_answers_GPT.to_csv('data/output/test_results_txt_to_sql_train_english_0613_test_multi.csv', index=False)
