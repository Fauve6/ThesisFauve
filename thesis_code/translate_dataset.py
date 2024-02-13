import jsonlines
import pandas as pd
import re


def load_data(file):
    question_line = []
    answer_line = []
    with jsonlines.open(file) as f:
        for line in f.iter():
            question_line.append(line['messages'][1]['content'])
            answer_line.append(line['messages'][2]['content'])

    return answer_line


def get_matches(regex_table, sql_code):
    matches = [x.group() for x in re.finditer(regex_table, sql_code, re.I)]
    select_items = []
    if len(matches) != 0:
        for match in matches:
            split_match = match.split()
            for split in split_match:
                select_items.append(str.lower(split))

    return select_items


if __name__ == "__main__":
    translate_to = "EN"

    sql_statements = load_data("data/test/test_everything_en.jsonl")
    translation_file = pd.read_excel("data/input_fauve_txt_to_sql_tables_and_columns_origin_and_translated_changed.xlsx")
    regex_join = r'(((?:LEFT JOIN)|(?:INNER JOIN)).+?((?=INNER)|(?=LEFT)|(?=WHERE)|(?=GROUP BY)|(?=ORDER BY)|' \
                   r'(?=LIMIT)|(?=HAVING)|(?=EXCEPT)|(?=UNION)|(?=INTERSECT)|(?=NESTED)|(?= ON)|$))'
    regex_from = r'((?:FROM).+?((?=INNER)|(?=LEFT)|(?=WHERE)|(?=GROUP BY)|(?=ORDER BY)|(?=LIMIT)|(?=HAVING)|(?=EXCEPT)|(?=UNION)|(?=INTERSECT)|(?=NESTED)|(?=\()|(?=\))|$))'
    translated_df = pd.DataFrame(columns=["translated_sql"])
    for sql_statement in sql_statements:
        translated_words = []
        word_list = sql_statement.split()
        join_tables = get_matches(regex_join, sql_statement)
        from_tables = get_matches(regex_from, sql_statement)
        filtered_from_tables_list = [word for word in from_tables if word.lower() != "from"]
        filtered_join_tables_list = [word for word in join_tables if word.lower() != "join"]
        filtered_join_tables_list = [word for word in filtered_join_tables_list if word.lower() != "inner"]
        filtered_join_tables_list = [word for word in filtered_join_tables_list if word.lower() != "left"]
        filtered_join_tables_list = [word for word in filtered_join_tables_list if word.lower() != "right"]

        tables = filtered_from_tables_list + filtered_join_tables_list

        translation_tables = pd.DataFrame()
        for table in tables:
            selection_translation_file = translation_file[translation_file['table_name'] == table.upper()]
            translation_tables = pd.concat([translation_tables,selection_translation_file], ignore_index=True)

        origin = translation_tables['origin'].tolist()
        for index, word in enumerate(word_list):
            if word.upper() in origin and translate_to == "EN":
                correct_row = translation_tables[translation_tables['origin'] == word.upper()]
                index_row = correct_row.index[0]
                if pd.notna(correct_row.loc[index_row, 'corrected_en']):
                    word_list[index] = correct_row.loc[index_row, 'corrected_en'].lower()
                else:
                    word_list[index] = correct_row.loc[index_row, 'translated_en'].lower()
            elif word.upper() in origin and translate_to == "NL":
                correct_row = translation_tables[translation_tables['origin'] == word.upper()]
                index_row = correct_row.index[0]
                if pd.notna(correct_row.loc[index_row, 'corrected_nl']):
                    word_list[index] = correct_row.loc[index_row, 'corrected_nl'].lower()
                else:
                    word_list[index] = correct_row.loc[index_row, 'translated_nl'].lower()
            elif word in tables and translate_to == "EN":
                correct_row = translation_tables[translation_tables['table_name'] == word.upper()]
                index_row = correct_row.index[0]
                word_list[index] = correct_row.loc[index_row, 'table_translation_en'].lower()
            elif word in tables and translate_to == "NL":
                correct_row = translation_tables[translation_tables['table_name'] == word.upper()]
                index_row = correct_row.index[0]
                word_list[index] = correct_row.loc[index_row, 'table_translation_nl'].lower()
        translated_string = ' '.join(word_list)
        new_row = {'translated_sql': translated_string}
        translated_df.loc[len(translated_df)] = new_row

    with jsonlines.open('data/test/test_everything_en.jsonl', 'r') as reader:
        with jsonlines.open('data/test/test_everything_english_translated_sql.jsonl', 'w') as writer:
            for line, new_value in zip(reader, translated_df['translated_sql']):
                line['messages'][2]['content'] = new_value

                writer.write(line)



