import pandas as pd
import sys
import re
from dotenv import load_dotenv
from pathlib import Path
from snowflake_connection import setup_connection
from snowflake.snowpark.types import StringType
from snowflake.snowpark.exceptions import SnowparkSQLException
from collections import Counter


def find_duplicate_columns(column_list):
    column_counts = Counter(column_list)
    duplicates = [column for column, count in column_counts.items() if count > 1]
    return duplicates


def get_GPT_answers(df_results):
    answers_gpt = df_results['answers_GPT3_5'].tolist()
    list_runnable_queries = []
    list_responses_gpt = []
    error = False

    for answer in answers_gpt:
        try:
            response = session.sql(answer)
            response.collect()
        except SnowparkSQLException as e:
            response = e.message
            list_responses_gpt.append(response)
            list_runnable_queries.append(0)
            error = True
        if not error:
            columns = [item[0] for item in response.dtypes]
            duplicate_columns = find_duplicate_columns(columns)
            if not duplicate_columns:
                list_runnable_queries.append(1)
                for column, datatype in response.dtypes:
                    if datatype == 'timestamp':
                        response_altered = response.withColumn(column, response[column].cast(StringType()))
                        response = response_altered
                correct_dataframe = response.toPandas()
                list_responses_gpt.append(correct_dataframe)
            else:
                list_responses_gpt.append("duplicate columns")
                list_runnable_queries.append(0)
        error = False

    return list_responses_gpt, list_runnable_queries


def get_golden_answers(df_results):
    answers_gold = df_results['answers_golden_standard'].tolist()
    list_responses_gold = []
    error = False
    for answer_g in answers_gold:
        try:
            response = session.sql(answer_g)
            response.collect()
        except SnowparkSQLException as e:
            response = e.message
            list_responses_gold.append(response)
            error = True
        if not error:
            for column, datatype in response.dtypes:
                if datatype == 'timestamp':
                    response_altered = response.withColumn(column, response[column].cast(StringType()))
                    response = response_altered
            correct_dataframe = response.toPandas()
            list_responses_gold.append(correct_dataframe)
        error = False
    return list_responses_gold


def calculate_valid_sql(nr_runnable_queries):
    total_runnable_queries = sum(nr_runnable_queries)

    score_valid_sql = total_runnable_queries / len(nr_runnable_queries)

    return score_valid_sql


def compare_tables(answers_gpt, answers_golden, queries_gold):
    check_equality_runnable = []
    check_equality_overall = []
    for index, answer_gold in enumerate(answers_golden):
        sorted_columns_gold = sorted(answer_gold.columns)
        answer_gold = answer_gold[sorted_columns_gold]

        if isinstance(answers_gpt[index], list) or isinstance(answers_gpt[index], str):
            check_equality_overall.append(0)
        else:
            sorted_columns_gpt = sorted(answers_gpt[index].columns)
            answer_gpt = answers_gpt[index][sorted_columns_gpt]

            if answer_gold.columns.str.lower().equals(answer_gpt.columns.str.lower()):
                answer_gpt.columns = answer_gpt.columns.str.lower()
                answer_gold.columns = answer_gold.columns.str.lower()
                if "ORDER BY" not in queries_gold[index] and "order by" not in queries_gold[index]:
                    colname = answer_gold.columns[0]
                    answer_gpt_sorted = answer_gpt.sort_values(by=colname).reset_index(drop=True)
                    answer_gold_sorted = answer_gold.sort_values(by=colname).reset_index(drop=True)
                    equality = answer_gpt_sorted.equals(answer_gold_sorted)
                else:
                    equality = answer_gpt.equals(answer_gold)
            else:
                equality = False

            if equality:
                check_equality_runnable.append(1)
                check_equality_overall.append(1)
            else:
                check_equality_runnable.append(0)
                check_equality_overall.append(0)

    return check_equality_runnable, check_equality_overall


def calculate_execution_accuracy(answers_gpt, answers_golden, data_df):
    golden_queries = data_df["answers_golden_standard"]
    equal_results_list_validSQL, equal_results_list_total = compare_tables(answers_gpt, answers_golden, golden_queries)

    total_equal_results_validSQL = sum(equal_results_list_validSQL)
    if total_equal_results_validSQL == 0:
        score_execution_accuracy_validSQL = 0.0
    else:
        score_execution_accuracy_validSQL = total_equal_results_validSQL / len(equal_results_list_validSQL)

    total_equal_results_total = sum(equal_results_list_total)
    if total_equal_results_total == 0:
        score_execution_accuracy_total = 0.0
    else:
        score_execution_accuracy_total = total_equal_results_total / len(equal_results_list_total)

    return score_execution_accuracy_validSQL, score_execution_accuracy_total


def get_select_items(select_regex, query):
    matches = [x.group() for x in re.finditer(select_regex,query, re.I)]
    select_items = []
    if len(matches) != 0:
        for match in matches:
            split_match = match.split()
            for split in split_match:
                select_items.append(str.lower(split))

        for index, item in enumerate(select_items):
            select_items[index] = item.replace(',', "")

    return select_items


def get_select_list(df_results, regex_select):
    answers_gold = df_results['answers_golden_standard'].tolist()
    answers_gpt = df_results['answers_GPT3_5'].tolist()

    list_select_items_gold = []
    list_select_items_gpt = []
    for sql_query in answers_gold:
        items_gold = get_select_items(regex_select, sql_query)
        list_select_items_gold.append(items_gold)
    for sql in answers_gpt:
        items_gpt = get_select_items(regex_select, sql)
        list_select_items_gpt.append(items_gpt)

    select_list_gold = []
    for index, row_gold in enumerate(list_select_items_gold):
        row_gpt = list_select_items_gpt[index]
        row_gpt.sort()
        row_gold.sort()
        if row_gpt == row_gold:
            select_list_gold.append(1)
        else:
            select_list_gold.append(0)

    return select_list_gold


def get_df_partial_pieces(df_results):
    select_regex = r'(?:SELECT).+(?=FROM)'
    select_list = get_select_list(df_results, select_regex)
    from_regex = r'((?:FROM).+?((?=INNER)|(?=LEFT)|(?=WHERE)|(?=GROUP BY)|(?=ORDER BY)|(?=LIMIT)|(?=HAVING)|(?=EXCEPT)|(?=UNION)|(?=INTERSECT)|(?=NESTED)|(?=\()|(?=\))|$))'
    from_list = get_select_list(df_results, from_regex)
    join_regex = r'((?:LEFT JOIN)|(?:INNER JOIN)).+?((?=INNER)|(?=LEFT)|(?=WHERE)|(?=GROUP BY)|(?=ORDER BY)|(?=LIMIT)|' \
                 r'(?=HAVING)|(?=EXCEPT)|(?=UNION)|(?=INTERSECT)|(?=NESTED)|$)'
    join_list = get_select_list(df_results, join_regex)
    where_regex = r'(?:WHERE.+?((?=ORDER BY)|(?=LIMIT)|(?=GROUP BY)|(?=UNION)|(?=HAVING)|$))'
    where_list = get_select_list(df_results, where_regex)
    group_by_regex = r'(?:GROUP BY.+?((?=ORDER BY)|(?=LIMIT)|(?=WHERE)|(?=UNION)|(?=HAVING)|$))'
    group_by_list = get_select_list(df_results, group_by_regex)
    order_by_regex = r'(?:ORDER BY.+?((?=GROUP BY)|(?=LIMIT)|(?=WHERE)|(?=UNION)|(?=HAVING)|$))'
    order_by_list = get_select_list(df_results, order_by_regex)
    limit_regex = r'(?:LIMIT.+?((?=GROUP BY)|(?=ORDER BY)|(?=WHERE)|(?=UNION)|(?=HAVING)|$))'
    limit_list = get_select_list(df_results, limit_regex)

    df_partial_pieces = pd.DataFrame({"select": select_list, "from": from_list, "join": join_list, 'where': where_list,
                                      'group_by': group_by_list, "order_by": order_by_list, "limit": limit_list})

    return df_partial_pieces


def calculate_exact_match(df_results):
    df = get_df_partial_pieces(df_results)
    df['exact_match'] = df.apply(lambda x: 0 if 0 in x.values else 1, axis=1)
    exact_match_list = df['exact_match'].tolist()
    print(df.loc[df['exact_match'] == 1])

    total_exact_matches = sum(exact_match_list)
    score_valid_sql = total_exact_matches / len(exact_match_list)
    return score_valid_sql


if __name__ == "__main__":
    # sys.path.append(Path.cwd().parent.parent)# mandatory adding of the parent directory, since python no longer
    # supports
    sys.path.insert(0, Path.cwd().parent.parent.as_posix())

    # load the .env variables, required since poetry doesn't load them by default (outside a notebook)
    load_dotenv()

    df_data = pd.read_csv('data/output/1106/test_results_txt_to_sql_train_english_1106_test_dutch.csv')

    connection_parameters_target = {}
    session = setup_connection(**connection_parameters_target)

    gpt_answers, runnable_queries = get_GPT_answers(df_data)
    golden_answers = get_golden_answers(df_data)
    valid_sql_score = calculate_valid_sql(runnable_queries)
    execution_accuracy_score_validSQL, execution_accuracy_score_total = calculate_execution_accuracy(gpt_answers,
                                                                                                     golden_answers, df_data)
    exact_match_score = calculate_exact_match(df_data)
    print(f"ValidSQL: {valid_sql_score}")
    print(f"EX_validSQL: {execution_accuracy_score_validSQL}")
    print(f"EX_total: {execution_accuracy_score_total}")
    print(f"EM_total: {exact_match_score}")
