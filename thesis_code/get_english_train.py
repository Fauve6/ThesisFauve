import jsonlines
import csv
import deepl
import re


# question_line = []
# with jsonlines.open('data/train/train_everything_en.jsonl') as f:
#     for line in f.iter():
#         question_line.append(line['messages'][1]['content'])

# translator = deepl.Translator(auth_key)
#
# translated_texts = []
# for question in question_line:
#     result = translator.translate_text(question, source_lang="NL", target_lang="EN-US")
#     translated_text = result.text
#     translated_text.append(translated_text)
#
# with open('english_train_questions.txt', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(translated_texts)

# with open('english_train_questions.txt', 'r') as file:
#     data = [line.rstrip('\n') for line in file.readlines()]
#
# question_line = []
# with jsonlines.open('data/train/train_everything_en.jsonl') as f:
#     for line in f.iter():
#         question_line.append(str(line))
#
# first_part_regex = r"^.+?(?:\'role\': \'user\', \'content\': \')"
# last_part_regex = r"(?<=\'user', 'content': ').+?$"
#
# list_new_strings = []
# for i, row in enumerate(question_line):
#     first_part = re.search(first_part_regex, row).group()
#     # [x.group() for x in re.finditer(first_part_regex,row, re.I)]
#     last_part = re.search(last_part_regex, row).group()
#     final_string = first_part + data[i] + last_part
#     list_new_strings.append(final_string)
#
# with jsonlines.open('data/train/train_everything_english.jsonl', mode='w') as writer:
#     writer.write_all(list_new_strings)
