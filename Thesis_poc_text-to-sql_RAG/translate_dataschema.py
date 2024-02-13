import pandas as pd
import re

def get_input():
    input = """
  """

    return input


if __name__ == "__main__":
    translate_to = "EN"

    to_translate = get_input()
    table = ""
    pattern = r'`([^`]*)`'
    translation_file = pd.read_excel("data/input_fauve_txt_to_sql_tables_and_columns_origin_and_translated_changed.xlsx")
    selection_translation_file = translation_file[translation_file['table_name'] == table.upper()]
    words_between_backticks = re.findall(pattern, to_translate)
    not_translated_words = re.findall(pattern, to_translate)
    origin = selection_translation_file['origin'].tolist()

    for index, word in enumerate(words_between_backticks):
        if word.upper() in origin and translate_to == "EN":
            correct_row = selection_translation_file[selection_translation_file['origin'] == word.upper()]
            index_row = correct_row.index[0]
            if pd.notna(correct_row.loc[index_row, 'corrected_en']):
                words_between_backticks[index] = correct_row.loc[index_row, 'corrected_en'].lower()
            else:
                words_between_backticks[index] = correct_row.loc[index_row, 'translated_en'].lower()
        elif word.upper() in origin and translate_to == "NL":
            correct_row = selection_translation_file[selection_translation_file['origin'] == word.upper()]
            index_row = correct_row.index[0]
            if pd.notna(correct_row.loc[index_row, 'corrected_nl']):
                words_between_backticks[index] = correct_row.loc[index_row, 'corrected_nl'].lower()
            else:
                words_between_backticks[index] = correct_row.loc[index_row, 'translated_nl'].lower()

    translated_string = to_translate
    for original, translated in zip(not_translated_words, words_between_backticks):
        translated_string = translated_string.replace(original, translated)

    print(translated_string)


