from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
import jsonlines
import pandas as pd
import os

model_name = "t5_model_finetuned_everything_en"
model_dir = f"data/model/experiments/{model_name}"

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

max_input_length = 512


def get_test_data():
    question_line = []
    answer_line = []
    with jsonlines.open('data/test/test_everything_en.jsonl') as f:
        for line in f.iter():
            question_line.append(line['messages'][1]['content'])
            answer_line.append(line['messages'][2]['content'])
    questions_df = pd.DataFrame(question_line, columns=["questions"])
    answers_df = pd.DataFrame(answer_line, columns=["answers"])

    return questions_df, answers_df


if os.path.isfile('data/output/t5/test_results_txt_to_sql_t5_everything_en.csv'):
    print("Test results already exist.")
else:
    df_questions, df_answers = get_test_data()
    questions = df_questions['questions'].tolist()

    answers_t5 = []
    for question in questions:
        inputs = ["translate question to SQL: " + question]
        inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, return_tensors="pt")
        output = model.generate(**inputs, num_beams=8, do_sample=True, min_length=10, max_length=64,
                                early_stopping=True)
        decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        answer_t5 = nltk.sent_tokenize(decoded_output.strip())[0]
        answers_t5.append(answer_t5)

    df_answers_t5 = pd.DataFrame(df_questions, columns=['questions'])
    df_answers_t5['answers_t5'] = answers_t5
    df_answers_t5['answers_golden_standard'] = df_answers['answers'].tolist()

    df_answers_t5.to_csv('data/output/t5/test_results_txt_to_sql_t5_everything_en.csv', index=False)
