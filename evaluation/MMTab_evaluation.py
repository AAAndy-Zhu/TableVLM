
import argparse
import json
import re
import tqdm
from collections import defaultdict
from sacrebleu.metrics import BLEU


# ## 1.Utility Functions

def convert_table_to_html_str(table_row_list=[]):
    """
    Given a list of table rows, build the corresponding html string, which is used to compute the TEDS score.
    We use the official code of PubTabNet to compute TEDS score, it does not consider '<th>' label.
    We also remove unneccessary spaces within a table cell and extra '\n' as they will influence the TEDS score.
    """
    html_table_str = "<html><body><table>" + '\n'
    for data_row in table_row_list:
        html_table_str += "<tr>"
        for cell_str in data_row:
            html_table_str += f"<td>{cell_str}</td>"
        html_table_str += "</tr>"
        html_table_str += '\n'
    html_table_str += "</table></body></html>"
    html_table_str = html_table_str.replace('\n','')
    return html_table_str

def convert_markdown_table_to_html(markdown_table):
    """
    Converts a markdown table to the corresponding html string for TEDS computation.
    """
    # remove extra code block tokens like '```markdown' and '```
    markdown_table = markdown_table.strip('```markdown').strip('```').strip() 
    row_str_list = markdown_table.split('\n')
    # extra the first header row and other data rows
    valid_row_str_list = [row_str_list[0]]+row_str_list[2:]
    table_rows = []
    for row_str in valid_row_str_list:
        one_row = []
        for cell in row_str.strip().split('|')[1:-1]:
            if set(cell) != set(' '):
                one_row.append(cell.strip())
            else:
                one_row.append(' ')
        table_rows.append(one_row)
    # build html string based on table rows
    html_str = convert_table_to_html_str(table_rows)
    return html_str

def convert_latex_table_to_html(latex_table):
    """
    Converts a markdown table to html string for TEDS computation.
    In the MMTab-eval, we only consider latex tables with similar structures of markdown tables.
    For other latex tables with compicated structures like merged cells, you need to rewrite this function to convert them.
    """
    # remove extra code block tokens like '```latex' and '```
    latex_table = latex_table.strip('```latex').strip('```').strip() 
    latex_table = latex_table.replace('\n', ' ')
    row_str_list = [row_str.strip('\n').strip('\\') for row_str in latex_table.split('\hline')[1:-1]]
    table_rows = []
    for row_str in row_str_list:
        one_row = []
        for c in row_str.split('&'):
            if set(c) != set(' '):
                one_row.append(c.strip())
            else:
                one_row.append(' ')
        table_rows.append(one_row)
    html_str = convert_table_to_html_str(table_rows)
    return html_str

def wrap_html_table(html_table):
    """
    The TEDS computation from PubTabNet code requires that the input html table should have <html>, <body>, and <table> tags.
    Add them if they are missing.
    """
    html_table = html_table.replace('\n','')
    # add missing <table> tag if missing
    if "<table" in html_table and "</table>" not in html_table:
        html_table = html_table + "</table>"
    elif "<table" not in html_table and "</table>" in html_table:
        html_table = "<table>" + html_table
    elif "<table" not in html_table and "</table>" not in html_table:
        html_table = "<table>" + html_table + "</table>"
    else:
        pass
    # add <body> and <html> tags if missing
    if '<body>' not in html_table:
        html_table = '<body>' + html_table + '</body>'
    if '<html>' not in html_table:
        html_table = '<html>' + html_table + '</html>'
    return html_table

# Read inference results of LLaVA model (merged.jsonl)
def read_llava_prediction_file(file_path):
    """
    Read LLaVA's inference results (e.g., merge.jsonl) and extract data of different benchmarks based on 'category' field.
    """
    predict_results = []
    with open(file_path, encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm.tqdm(lines):
            item = json.loads(line.strip())
            if isinstance(item['prediction'], list):
                item['prediction'] = item['prediction'][0]
            predict_results.append(item)
    # print("Predicted Sample Number:",len(predict_results))
    benchmark_name_to_predicted_item_list = defaultdict(list)
    for item in predict_results:
        if 'dataset_name' not in item:
            dataset_name = item['category'].split('_')[0]
            task_name = item['category'].split('_')[2]
        else:
            dataset_name = item['dataset_name'] # e.g., TabFact
            task_name = item['task_type'] # e.g., TFV
        # for table structure understanding tasks, benchmark name is the task name
        if task_name not in ['TSD','TCL','RCE','MCD','TCE','TR','OOD_TSD','OOD_TCL','OOD_RCE','OOD_TCE']:
            benchmark_name = dataset_name
        else:
            benchmark_name = task_name
        benchmark_name_to_predicted_item_list[benchmark_name].append(item)
    for benchmark_name,  predicted_item_list in benchmark_name_to_predicted_item_list.items():
        item_num = len(predicted_item_list)
        # print(f'benchmark name: {benchmark_name}, test data num: {item_num}')
    return benchmark_name_to_predicted_item_list


# print(MMTab_eval_test_data[0])

# print(table_id_to_test_table['TABMWP_8'])
# exit()

# ## 3.Evaluation Functions

# ### 3.1 TQA, TFV and T2T Tasks

def extract_tqa_answer_list(model_output):
    """
    Extract the answer list from the model output to compute accuracy
    """
    model_output = model_output.replace('\n',' ')
    ret = re.match('.*({[\"\']answer[\"\']\:.*}).*',model_output)
    if ret is not None:
        answer_str = ret.group(1)
        try:
            answer_str = re.sub('[\"\']+',"\"",answer_str)
            answer_item = eval(answer_str)
            predicted_answer = answer_item['answer']
            if type(predicted_answer) != list and type(predicted_answer) == str:
                predicted_answer = [predicted_answer]
            elif type(predicted_answer) != list and type(predicted_answer) in [float,int]:
                predicted_answer = [str(predicted_answer)]
            else:
                pass
        # The answer is considered to be wrong if we can not extract answer list from the json str
        except:
            predicted_answer = []
        return predicted_answer
    else:
        return []

def evaluate_tqa_questions(benchmark_name,pred_item_list):
    """
    Evaluation for table question answering (TQA) and table fact verification (TFV) benchmark.
    Metric: accuracy.
    Note that some baseline models can not strictly follow instructions to output the final answer in the required JSON format.
    For instance, Qwen-VL may only output a short answer due to the potential overfitting of training data.
    In such cases, the evaluation script needs to be changed according to the characteristic of certain model output.
    """
    correct_item_list = []
    wrong_item_list = []
    failed_item_list = []
    for item in pred_item_list:
        try:
            # item_id = item['question_id']
            # ori_item = item_id_to_test_item[item_id]
            model_output = item['prediction']
            # parse the predicted answer list
            predicted_answer_list = extract_tqa_answer_list(model_output)
            gold_answer_list = item['answer_list']
            # Sometimes the order of multiple answer text is not necessarily same as the gold answer,
            # so we convert the answer list to a set for comparison
            if set(gold_answer_list) == set(predicted_answer_list):
                correct_item_list.append(item)
            else:
                wrong_item_list.append(item)
        except Exception:
            failed_item_list.append(item)
            
    print("Benchmark: ",benchmark_name)
    correct_num = len(correct_item_list)
    total_sample_num = len(pred_item_list)
    print("Accuracy: ", correct_num/total_sample_num)
    # print(f"{benchmark_name}:", correct_num/total_sample_num * 100)
    # print(correct_num/total_sample_num * 100)


    problem_sample_num = len(failed_item_list)
    print("Total sample number:",total_sample_num)
    print(f"There are {problem_sample_num} samples that failed to be evaluated.")
    print("-"*20)
    return correct_item_list, wrong_item_list

def evaluate_tabmcq_questions(benchmark_name,pred_item_list):
    """
    Evaluation for TabMCQ benchmark.
    Metric: accuracy. 
    """
    correct_item_list = []
    wrong_item_list = []
    failed_item_list = []
    
    for item in pred_item_list:
        try:
            # item_id = item['question_id']
            # ori_item = item_id_to_test_item[item_id]
            model_output = item['prediction']
            model_output = model_output.replace('\n',' ')
            model_output = model_output.replace(']','')
            model_output = model_output.replace('[','')
            ret = re.match('.*({[\"\']answer[\"\']\:\s?.*?}).*',model_output)
            # parse predicted answer
            if ret is not None:
                answer_str = ret.group(1)
                answer_item = eval(answer_str)
                predicted_answer = answer_item['answer']
                if type(predicted_answer) == list:
                    predicted_answer = predicted_answer[0]
                gold_answer_str = item['answer_list'][0] # e.g., '(D) Blood'
                # Sometimes the predicted answer does not contain option letter like '(D)'
                # To deal with such cases, we also consider removing the option letter in ground truth for comparison
                if predicted_answer == gold_answer_str or predicted_answer == ' '.join(gold_answer_str.split(' ')[1:]):
                    correct_item_list.append(item)
                else:
                    wrong_item_list.append(item)
            else:
                failed_item_list.append(item)
        except Exception:
            failed_item_list.append(item)
            
    print(f"Benchmark: {benchmark_name}")
    total_sample_num = len(pred_item_list)
    correct_num = len(correct_item_list)
    print("Accuracy: ",correct_num/total_sample_num)
    # print(correct_num/total_sample_num * 100)
    # print(f"{benchmark_name}:", correct_num/total_sample_num * 100)

    problem_sample_num = len(failed_item_list)
    print("Total sample number:",total_sample_num)
    print(f"There are {problem_sample_num} samples that failed to be evaluated.")
    print("-"*20)

def evaluate_text_generation_questions(benchmark_name,pred_item_list):
    """
    Evaluation for table-to-text benchmark.
    Metric: bleu.
    More metrics like ROUGE or LLM-as-a-judge rating are needed for a more robust evaluation.
    """
    bleu = BLEU()
    output_text_list = [] # output text 
    reference_text_list = [] # reference text list
    for item in pred_item_list:
        pred_text = item['prediction']
        # item_id = item['question_id']
        # ori_item = item_id_to_test_item[item_id]
        gold_text = item['output']
        assert gold_text not in ['','None']
        output_text_list.append(pred_text)
        reference_text_list.append(gold_text)
    assert len(output_text_list) == len(reference_text_list)
    bleu_score = bleu.corpus_score(output_text_list, [reference_text_list])
    print("Benchmark: ",benchmark_name)
    print("BLEU score:",bleu_score)
    print("-"*20)


# ### 3.2 Table Structure Understanding (TSU) Tasks

# Initialize TEDS object, 'n_jobs' is the number of parallel threads 

def evaluate_mcd_questions(benchmark_name,pred_item_list):
    """
    Evaluation for merged cell detection (MCD) benchmark.
    Metric: precision, recall and F1 score
    """
    pred_cell_num = 0 # number of predicted merged cells 
    gold_cell_num = 0 # number of gold merged cells
    correct_cell_num = 0 # number of predicted merged cells which are correct
    failed_item_list = []
    for item in pred_item_list:
        try:
            # item_id = item['question_id']
            # ori_item = item_id_to_test_item[item_id]
            model_output = item['prediction']
            model_output = model_output.replace('\n',' ')
            gold_answer_list = item['answer_list']
            merged_cell_region_list = []
            gold_cell_num += len(gold_answer_list)
            if gold_answer_list == ['None']: # There are no merged cells in the table
                # Different models may use different sentences to express 'there is no merged cell'.
                # In such cases, you need to include more expressions for a more accurate evaluation.
                if "does not contain any merged cells" in model_output.lower() or "no merged cell" in model_output.lower():
                    correct_cell_num += 1
                    pred_cell_num += 1
                else:
                    pred_cell_num += 1
            else:  # There are merged cells in the table
                # parse the ground truth coordinates of merged cells
                for answer_str in gold_answer_list:
                    gold_answer_item = eval(answer_str)
                    top_row_id, left_col_id = gold_answer_item['top-left']
                    bottom_row_id, right_col_id = gold_answer_item['bottom-right']
                    gold_merged_region_repr = f"{top_row_id}_{left_col_id}_{bottom_row_id}_{right_col_id}"
                    merged_cell_region_list.append(gold_merged_region_repr)
                # parse the predicted coordinates of merged cells
                pred_answer_str_list = re.findall('{[\"\']top-left[\"\']\:.*?,\s?[\"\']bottom-right[\"\']\:.*?}',model_output)
                for answer_str in pred_answer_str_list:
                    pred_answer_item = eval(answer_str)
                    top_row_id, left_col_id = pred_answer_item['top-left']
                    bottom_row_id, right_col_id = pred_answer_item['bottom-right']
                    pred_merged_region_repr = f"{top_row_id}_{left_col_id}_{bottom_row_id}_{right_col_id}"
                    if pred_merged_region_repr in merged_cell_region_list:
                        correct_cell_num += 1
                pred_cell_num += len(pred_answer_str_list)
        except Exception as e:
            failed_item_list.append(item)
            item['exception'] = e
             
    print(f"Benchmark: {benchmark_name}")
    P = correct_cell_num / pred_cell_num
    R = correct_cell_num / gold_cell_num
    print("Precision:",P)
    print("Recall:",R)
    if P+R == 0:
        F1 = 0
    else:
        F1 = 2*P*R/(P+R)
    print("F1 score:",F1)
    total_sample_num = len(pred_item_list)
    problem_sample_num = len(failed_item_list)
    print("Total sample number:",total_sample_num)
    print(f"There are {problem_sample_num} samples that failed to be evaluated.")
    print("-"*20)
    # print(f"{benchmark_name}:", F1)

def evaluate_tcl_questions(benchmark_name,pred_item_list):
    """
    Evaluation for table cell locating (TCL) benchmark.
    Metric: cell-level accuracy
    """
    total_cell_num = 0
    correct_cell_num = 0
    failed_item_list = []
    for item in pred_item_list:
        try:
            # item_id = item['question_id']
            # ori_item = item_id_to_test_item[item_id]
            model_output = item['prediction']
            model_output = model_output.replace('\n',' ')
            model_output = model_output.replace('\\','')
            gold_output = item['output']
            # parse the ground truth cell locations (row_id, column_id)
            # example gold_dict_str_list = [('Raúl Hidalgo', '(13, 1)'),
            #                               ('Year', 'DOES NOT EXIST')],
            gold_dict_str_list = re.findall('{[\"\']value[\"\']\:\s?[\"\'](.*?)[\"\'],\s?[\"\']location[\"\']\:\s?[\"\']?(.*?)[\"\']?}',gold_output)
            cell_str_to_location = {}
            for cell_str,location_str in gold_dict_str_list:
                cell_str_to_location[cell_str] = location_str
            total_cell_num += len(cell_str_to_location)
            item['cell_str_to_gold_location'] = cell_str_to_location
            # parse the predicted cell locations
            pred_dict_str_list = re.findall('{[\"\']value[\"\']\:\s?[\"\'](.*?)[\"\'],\s?[\"\']location[\"\']\:\s?[\"\']?(.*?)[\"\']?}',model_output)
            cell_str_to_pred_location = {}
            for cell_str,location_str in pred_dict_str_list:
                if cell_str in cell_str_to_location:
                    gold_cell_location = cell_str_to_location[cell_str]
                    pred_cell_location = location_str
                    cell_str_to_pred_location[cell_str] = location_str
                    if str(gold_cell_location).lower() == str(pred_cell_location).lower():
                        correct_cell_num += 1 
            item['cell_str_to_pred_location'] = cell_str_to_pred_location
                
        except Exception as e:
            failed_item_list.append(item)
            item['exception'] = e
            
    print(f"Benchmark: {benchmark_name}")
    print("Cell-level accuracy:",correct_cell_num/total_cell_num)
    total_sample_num = len(pred_item_list)
    problem_sample_num = len(failed_item_list)
    print("Total sample number:",total_sample_num)
    print(f"There are {problem_sample_num} samples that failed to be evaluated.")
    print("-"*20)
    # print(f"{benchmark_name}:", correct_cell_num/total_cell_num)
    
def evaluate_rce_questions(benchmark_name,pred_item_list):
    """
    Evaluation for row and column extraction (RCE) benchmark.
    Metric: row and column level F1 score
    """
    row_correct_cell_num = 0 
    row_pred_cell_num = 0 
    row_ori_cell_num = 0 
    column_correct_cell_num = 0 
    column_pred_cell_num = 0 
    column_ori_cell_num = 0
    
    failed_item_list = []
    
    for item in pred_item_list:
        try:
            # item_id = item['question_id']
            # ori_item = item_id_to_test_item[item_id]
            image_id = item['image_id']
            table_rows = table_id_to_test_table[image_id]['table_rows']
            model_output = item['prediction']
            
            row_id_to_correct_cell_list = {}
            col_id_to_correct_cell_list = {}
            # parse the predicted cells of specific row or column
            match_group_tuple_list = re.findall('{[\"\']row_id[\"\']\:(.*?),\s?[\"\']cell_list[\"\']\:(.*?)}|{[\"\']column_id[\"\']\:(.*?),\s?[\"\']cell_list[\"\']\:(.*?)}',model_output)
            for matched_tuple in match_group_tuple_list:
                if matched_tuple[0] != '': # extract cells from a specific row
                    row_id = int(eval(matched_tuple[0]))
                    pred_cell_list = eval(matched_tuple[1])
                    target_cell_list = table_rows[row_id-1] # the ground truth cells in the original row
                    row_pred_cell_num += len(pred_cell_list)
                    row_ori_cell_num += len(target_cell_list)
                    correct_cell_list = [c for c in pred_cell_list if c in target_cell_list] # predicted cells that are also in the ground truth
                    row_correct_cell_num += len(correct_cell_list)
                    row_id_to_correct_cell_list[row_id] = correct_cell_list
                else: # extract cells from a specific column
                    column_id = int(eval(matched_tuple[2]))
                    pred_cell_list = eval(matched_tuple[3])
                    target_cell_list = [] # the ground truth cells in the original column  
                    for row in table_rows:
                        if len(row) == 1:
                            target_cell_list.append(row[0])
                        else:
                            target_cell_list.append(row[column_id-1])
                    column_pred_cell_num += len(pred_cell_list)
                    column_ori_cell_num += len(target_cell_list)
                    correct_cell_list = [c for c in pred_cell_list if c in target_cell_list] # predicted cells that are also in the ground truth
                    column_correct_cell_num += len(correct_cell_list)
                    col_id_to_correct_cell_list[column_id] = correct_cell_list
            item['row_id_to_correct_cell_list'] = row_id_to_correct_cell_list
            item['col_id_to_correct_cell_list'] = col_id_to_correct_cell_list
            
        except Exception as e:
            failed_item_list.append(item)
            item['exception'] = e
            
    print(f"Benchmark: {benchmark_name}")
    row_P = row_correct_cell_num/row_pred_cell_num # row-level precision
    row_R = row_correct_cell_num/row_ori_cell_num # row-level recall
    row_F1 = 2*row_P*row_R/(row_P+row_R) # row-level F1
    col_P = column_correct_cell_num/column_pred_cell_num if  column_pred_cell_num > 0 else 0  # column-level precision
    col_R = column_correct_cell_num/column_ori_cell_num if column_ori_cell_num > 0 else 0  # column-level recall
    col_F1 = 2*col_P*col_R/(col_P+col_R) if col_P+col_R > 0 else 0 # column-level F1
    
    print("Row-level Precision:",row_P)
    print("Row-level Recall:",row_R)
    print("Row-level F1:",row_F1)
    print("")
    print("Column-level Precision:",col_P)
    print("Column-level Recall:",col_R)
    print("Column-level F1:",col_F1)
    
    total_sample_num = len(pred_item_list)
    problem_sample_num = len(failed_item_list)
    print("Total sample number:",total_sample_num)
    print(f"There are {problem_sample_num} samples that failed to be evaluated.")
    print("-"*20)
    # print(f"{benchmark_name}_row:", row_F1)
    # print(f"{benchmark_name}_column:", col_F1)

def evaluate_tce_questions(benchmark_name,pred_item_list):
    """
    Evaluation for table cell extraction (TCE) benchmark.
    Metric: cell-level accuracy
    """
    total_cell_num = 0
    correct_cell_num = 0
    failed_item_list = []
    for item in pred_item_list:
        try:
            # item_id = item['question_id']
            # ori_item = item_id_to_test_item[item_id]
            model_output = item['prediction']
            model_output = model_output.replace('\n',' ')
            gold_output = item['output']
            # parse ground truth cell value
            gold_dict_str_list = re.findall('{[\"\']row_id[\"\']\:.*?[\"\']column_id[\"\']\:.*?[\"\']cell_value[\"\']\:.*?}',gold_output)
            cell_location_to_cell_str = {}
            for dict_str in gold_dict_str_list:
                cell_item = eval(dict_str)
                row_id = cell_item['row_id']
                col_id = cell_item['column_id']
                gold_cell_value = cell_item['cell_value']
                cell_location_to_cell_str[f"{row_id}_{col_id}"] = gold_cell_value
            total_cell_num += len(cell_location_to_cell_str)
            # parse predicted cell value
            pred_dict_str_list = re.findall('{[\"\']row_id[\"\']\:.*?[\"\']column_id[\"\']\:.*?[\"\']cell_value[\"\']\:.*?}',model_output)
            cell_location_to_pred_str = {}
            for dict_str in pred_dict_str_list:
                # some output may contain extra '[' or ']'
                dict_str = dict_str.replace(']','')
                dict_str = dict_str.replace('[','')
                cell_item = eval(dict_str)
                row_id = cell_item['row_id']
                col_id = cell_item['column_id']
                cell_location = f"{row_id}_{col_id}"
                if (cell_location in cell_location_to_cell_str) and (cell_location not in cell_location_to_pred_str) :
                    gold_cell_value = cell_location_to_cell_str[cell_location]
                    pred_cell_value = cell_item['cell_value']
                    cell_location_to_pred_str[cell_location] = pred_cell_value
                    if str(pred_cell_value).lower() == str(gold_cell_value).lower():
                        correct_cell_num += 1
    
        except Exception as e:
            failed_item_list.append(item)
            item['exception'] = e
            
    print(f"Benchmark: {benchmark_name}")
    print("cell level accuracy: ",correct_cell_num/total_cell_num)
    total_sample_num = len(pred_item_list)
    problem_sample_num = len(failed_item_list)
    print("Total sample number:",total_sample_num)
    print(f"There are {problem_sample_num} samples that failed to be evaluated.")
    print("-"*20)
    # print(f"{benchmark_name}:", correct_cell_num/total_cell_num)

def evaluate_tsd_questions(benchmark_name,pred_item_list):
    """
    Evaluation for table size detection (TSD) benchmark.
    Metric: row number and column number accuracy
    """
    row_correct_num = 0
    col_correct_num = 0
    failed_item_list = []
    
    for item in pred_item_list:
        try:
            # item_id = item['question_id']
            # ori_item = item_id_to_test_item[item_id]
            model_output = item['prediction'].lower()
            # print("model_output:",model_output)
            model_output = model_output.replace('\n',' ')
            model_output = model_output.replace('\\','')
            # parse predicted row number and column number
            if 'row_number' in model_output:
                ret = re.match('.*({.*[\"\']row_number[\"\']:.*[\"\']column_number[\"\']:.*}).*',model_output)
                answer_str = ret.group(1)
                answer_item = eval(answer_str) 
                pred_row_number = answer_item['row_number']
                pred_col_number = answer_item['column_number']
            else:
                ret = re.match('.*(\d+) rows and (\d+) columns.*',model_output)
                pred_row_number = ret.group(1)
                pred_col_number = ret.group(2)
            # extract ground truth row number and column number
            gold_answer_tuple = item['answer_list'][0]
            gold_row_number = str(gold_answer_tuple[0])
            gold_col_number = str(gold_answer_tuple[1])
            if pred_row_number == gold_row_number:
                row_correct_num += 1
            if pred_col_number == gold_col_number:
                col_correct_num += 1

        except Exception as e:
            item['exception'] = e
            failed_item_list.append(item)
            
    print(f"Benchmark: {benchmark_name}")
    total_sample_num = len(pred_item_list)
    print("row number accuracy:",row_correct_num/total_sample_num)
    print("column number accuracy:",col_correct_num/total_sample_num)
    problem_sample_num = len(failed_item_list)
    print("Total sample number:",total_sample_num)
    print(f"There are {problem_sample_num} samples that failed to be evaluated.")
    print("-"*20)
    # print(f"{benchmark_name}_row:", row_correct_num/total_sample_num)
    # print(f"{benchmark_name}_column:", col_correct_num/total_sample_num)


DEFAULT_BENCHMARKS = [
    'TSD', 'TCL', 'RCE', 'MCD', 'TCE',
    'OOD_TSD', 'OOD_TCE', 'OOD_TCL', 'OOD_RCE',
    'WTQ', 'HiTab', 'TAT-QA', 'AIT-QA', 'TabMCQ',
    'TabFact', 'InfoTabs', 'PubHealthTab',
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MMTab prediction results.")
    parser.add_argument(
        '--prediction_file',
        type=str,
        required=True,
        help='Path to the prediction JSONL file.',
    )
    parser.add_argument(
        '--eval_data_file',
        type=str,
        required=True,
        help='Path to MMTab eval test data JSON file.',
    )
    parser.add_argument(
        '--eval_tables_file',
        type=str,
        required=True,
        help='Path to MMTab eval test tables JSON file.',
    )
    parser.add_argument(
        '--benchmarks',
        type=str,
        default=','.join(DEFAULT_BENCHMARKS),
        help='Comma-separated benchmark names to evaluate.',
    )
    return parser.parse_args()


def main():
    global item_id_to_test_item
    global table_id_to_test_table

    args = parse_args()

    # read the predicted data
    benchmark_name_to_predicted_item_list = read_llava_prediction_file(args.prediction_file)

    # read the ground truth data
    with open(args.eval_data_file, encoding='utf-8') as f:
        MMTab_eval_test_data = json.load(f)
    # item_id --> test data
    item_id_to_test_item = {}
    for item in MMTab_eval_test_data:
        item_id = item['item_id']
        item_id_to_test_item[item_id] = item
    print("MMTab-eval data num: ", len(MMTab_eval_test_data))

    # table_id --> test table
    with open(args.eval_tables_file, encoding='utf-8') as f:
        MMTab_eval_test_tables = json.load(f)
    table_id_to_test_table = {}
    for table_item in MMTab_eval_test_tables:
        table_id = table_item['image_id']
        table_id_to_test_table[table_id] = table_item
    print("MMTab-eval table num: ", len(table_id_to_test_table))

    benchmark_name_list = [name.strip() for name in args.benchmarks.split(',') if name.strip()]
    all_correct_items = []
    all_wrong_items = []
    for benchmark_name in benchmark_name_list:
        if benchmark_name not in benchmark_name_to_predicted_item_list:
            print(f"Skip benchmark: {benchmark_name} as there is no predicted results.")
            print("-"*20)
            continue
        predicted_item_list = benchmark_name_to_predicted_item_list[benchmark_name]
        if benchmark_name in ['TSD','OOD_TSD']:
            evaluate_tsd_questions(benchmark_name,predicted_item_list)
        elif benchmark_name in ['TCE','OOD_TCE']:
            evaluate_tce_questions(benchmark_name,predicted_item_list)
        elif benchmark_name in ['TCL','OOD_TCL']:
            evaluate_tcl_questions(benchmark_name,predicted_item_list)
        elif benchmark_name in ['RCE','OOD_RCE']:
            evaluate_rce_questions(benchmark_name,predicted_item_list)
        elif benchmark_name == 'MCD':
            evaluate_mcd_questions(benchmark_name,predicted_item_list)
        elif benchmark_name == 'TabMCQ':
            evaluate_tabmcq_questions(benchmark_name,predicted_item_list)
        elif benchmark_name in ['FeTaQA','HiTab_t2t','Rotowire','WikiBIO']:
            evaluate_text_generation_questions(benchmark_name,predicted_item_list)
        else:
            correct_item_list, wrong_item_list = evaluate_tqa_questions(benchmark_name,predicted_item_list)
            all_correct_items.extend(correct_item_list)
            all_wrong_items.extend(wrong_item_list)


if __name__ == "__main__":
    main()
