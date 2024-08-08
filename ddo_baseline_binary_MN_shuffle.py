import os
import time
import openai
import json
import random
from tqdm import tqdm
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORGANIZATION_KEY")


def count_tokens(text):
    return len(text.split())

def get_completion_from_messages(messages,
                                 model='gpt-3.5-turbo-16k-0613',  # gpt-3.5-turbo-0613
                                 temperature=0,  # higher value means higher uncertainty of the replies
                                 max_tokens=480):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        presence_penalty=0.7,
        max_tokens=max_tokens,
    )
    return {
        "total_tokens": response["usage"]["total_tokens"],
        "completion_tokens": response["usage"]["completion_tokens"],
        "content": response.choices[0]["message"]["content"],
    }



if __name__ == "__main__":
    
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--seed', type=int, default=12345)
    argparser.add_argument('--debate_num', type=int, default=500)
    argparser.add_argument('--model', type=str, default='gpt-3.5', choices=['gpt-3.5', 'gpt-4'])
    argparser.add_argument('--vote_diff', type=int, default=2, choices=[2, 3, 4, 5])
    argparser.add_argument('--file', type=str, default=None, help='The file which contains the debate names to work on')
    argparser.add_argument('--new_instruction', action='store_true', default=False, help='Whether to use the new instruction')
    argparser.add_argument('--rounds', type=int, default=0, help='The number of rounds of the debate', choices=[3, 4, 5])
    args = argparser.parse_args()
    
    if args.file is None:
        assert args.rounds != 0, "Please provide the number of rounds of the debate"
    
    if args.model == 'gpt-3.5':
        model_name = "gpt-3.5-turbo-1106" #"gpt-3.5-turbo-16k-0613"
        MAX_EXECUTIONS_PER_MINUTE = 3500
        MAX_TOKENS_PER_MINUTE = 150000
        
    else:
        model_name = "gpt-4-1106-preview"
        MAX_EXECUTIONS_PER_MINUTE = 200
        MAX_TOKENS_PER_MINUTE = 10000

    with open("debates.json", "r") as f:
        debates = json.load(f)


    # the following saves the debate with its corresponding results
    debate_results = {}
    
    for topic, debate in debates.items():
        if debate['forfeit_label']:
            continue
        if debate['number_of_rounds'] != args.rounds:
            continue

        flag = False
        for round in debate['rounds']:
            for side in round:
                if side['text'] == 'forfeit':
                    flag = True
                    break
            if flag:
                break
        if flag:
            continue

        pro_side = debate['participant_1_name']
        con_side = debate['participant_2_name']
        pro_votes = 0
        con_votes = 0
        for vote in debate['votes']:
            if 'Made more convincing arguments' in vote['votes_map'][pro_side]:
                pro_votes += vote['votes_map'][pro_side]['Made more convincing arguments']
                con_votes += vote['votes_map'][con_side]['Made more convincing arguments']
        
        if pro_votes > con_votes+args.vote_diff:
            debate_results[topic] = 1
        elif pro_votes+args.vote_diff < con_votes:
            debate_results[topic] = -1
    print('There are ', len(debate_results), f' debates in {args.rounds} round with greater than {args.vote_diff} difference in votes.')

    file_prefix = 'debate_pred_binary'
    # if proviede a file, the use the debates from the file
    if args.file:
        file_prefix = args.file.split('.')[0]
        with open(args.file, 'r') as f:
            # remove the "\n" token at the end of each line
            selected_debate_names = [line.strip() for line in f.readlines()]

    else:
        available_debates = []
        
        with open("debate_lens.json", "r") as f:
            debate_lens = json.load(f)

        for topic, result in debate_results.items():
            debate_len = int(debate_lens[topic]) + 100

            if debate_len < 10000:
                available_debates.append(topic)

        debate_names = sorted(available_debates)
        args.debate_num = len(debate_names)

        random.seed(args.seed)
        if len(debate_names) < args.debate_num:
            print('There are not enough debates.', len(debate_names))
            selected_debate_names = debate_names
            args.debate_num = len(debate_names)
        else:
            selected_debate_names = random.sample(debate_names, args.debate_num)
        file_prefix += f'_{args.vote_diff}_{args.rounds}'

    # get the ground truth labels for these debates
    gt_labels = {debate_name: debate_results[debate_name]
                 for debate_name in selected_debate_names}
    
    existing_predictions = {}

    if os.path.exists(f'experiments/{file_prefix}_{model_name}_{args.seed}_{args.debate_num}_remove_forfeit_change_names_shuffled-MN.json'):
        with open(f'experiments/{file_prefix}_{model_name}_{args.seed}_{args.debate_num}_remove_forfeit_change_names_shuffled-MN.json', 'r') as f:
            for line in f:
                line_data = json.loads(line.strip())
                existing_predictions.update(line_data)

    executions_this_minute = 0
    tokens_this_minute = 0
    last_reset_time = time.time()
    
    dir_name = f'{file_prefix}_{model_name}_remove_forfeit_change_names_shuffled-MN_log'
    
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    
    debate_pred = {}
    print(len(selected_debate_names), len(existing_predictions))
    for debate_name in selected_debate_names:
            
        if debate_name in existing_predictions:
            debate_pred[debate_name] = existing_predictions[debate_name]
            continue
        
        masked_name = {0: 'M', 1: 'N'}

        current_debate = debates[debate_name]
        context = ''
        title = current_debate['title']
        ## shuffle the label verbalizer
        # pro_index = random.choice([0, 1])
        ## fixed label verbalizer
        pro_index = 0

        con_index = 1 - pro_index
        indices = {'Pro': pro_index, 'Con': con_index}
        
        
        prob = random.uniform(0, 1) # shuffle the position
        # prob = 1 # always starts with asking Pro side
        # prob = 0 # always starst with asking Con side
        if prob > 0.5:
            # instruction =   f"Assume you are a debate evaluator, there are two participants in this debate. Given the above context of the whole debate. \
            #         please give the decision on which participant is the winner, \
            #         you only need to give the character of either A, or B. A means user {masked_name[pro_index]} wins. B means user {masked_name[con_index]} wins. \
            #         Please only give the result without any other words"
            instruction =   f"Assume you are a debate evaluator, there are two participants in this debate. Given the above context of the whole debate. \
                    please give the decision on which participant is the winner, \
                    you only need to give the character of either {masked_name[pro_index]}, or {masked_name[con_index]}. {masked_name[pro_index]} means user {masked_name[pro_index]} wins. {masked_name[con_index]} means user {masked_name[con_index]} wins. \
                    Please only give the result without any other words."
            new_instruction =  f'Assume you are the debate evaluator, there are two participants in this debate, \
                    after given the context of the debate, please give the decision on which participant is the winner in terms of making more convincing arguments, \
                    without considering the grammar and spelling, the reliability of the source and the debater conduct. \
                    You only need to give the character of either {masked_name[pro_index]} or {masked_name[con_index]}, {masked_name[pro_index]} means the user {masked_name[pro_index]} wins, {masked_name[con_index]} means user {masked_name[con_index]} wins. \
                    Please only give the result without any other words.'
        else:
            instruction =   f"Assume you are a debate evaluator, there are two participants in this debate. Given the above context of the whole debate. \
                    please give the decision on which participant is the winner, \
                    you only need to give the character of either {masked_name[con_index]}, or {masked_name[pro_index]}. {masked_name[con_index]} means user {masked_name[con_index]} wins.  {masked_name[pro_index]} means user {masked_name[pro_index]} wins. \
                    Please only give the result without any other words"
            new_instruction =  f'Assume you are the debate evaluator, there are two participants in this debate, \
                    after given the context of the debate, please give the decision on which participant is the winner in terms of making more convincing arguments, \
                    without considering the grammar and spelling, the reliability of the source and the debater conduct. \
                    You only need to give the character of either {masked_name[con_index]} or {masked_name[pro_index]}, {masked_name[con_index]} means the user {masked_name[con_index]} wins, {masked_name[pro_index]} means user {masked_name[pro_index]} wins. \
                    Please only give the result without any other words.'   
        
        for i, round in enumerate(current_debate['rounds']):
            for side in round:
                cur_side = side['side']
                if cur_side == 'Cry Me a River,': # an issue with the original data
                    cur_side = 'Con'
                if i == 0:
                    try:
                        context += 'The current speech in the debate is from user ' + masked_name[indices[cur_side]] + ': ' \
                            + f'I am the {cur_side} side of this debate, {title}. ' + side['text'] + ' \n\n'
                    except:
                        context = "the current debate contains some error, please directly predict it with 3 without saying other words"
                        continue
                else:
                    context += 'The current speech in the debate is from user ' + masked_name[indices[cur_side]] + ': ' + side['text'] + ' \n\n'

        if not args.new_instruction:
            combined_prompt =  "Debate content:\n" + context \
                + "\n\nInstruction:\n" + instruction
        else:
            combined_prompt = "Debate content:\n" + context \
                + "\n\nInstruction:\n" + new_instruction
 
        tokens_per_minute = count_tokens(combined_prompt)


        current_time = time.time()
        time_elapsed = current_time - last_reset_time
        
        
        if time_elapsed >= 60:
            last_reset_time = current_time
            executions_this_minute = 0
            tokens_this_minute = 0
        
        if executions_this_minute + 100 >= MAX_EXECUTIONS_PER_MINUTE or tokens_this_minute + tokens_per_minute + 3000 > MAX_TOKENS_PER_MINUTE:
            sleep_time = 5  # sleep until the next minute
            time.sleep(sleep_time)
            last_reset_time = time.time()
            executions_this_minute = 0
            tokens_this_minute = 0
        
        
        executions_this_minute += 1

        messages = [
            {
                'role': 'user',
                'content': combined_prompt
            },
        ]

    
        response = None
        try:
            response = get_completion_from_messages(messages, model=model_name)
            if response['content'].split()[-1].strip('() ') == masked_name[pro_index]:
                score = 1
            elif response['content'].split()[-1].strip('() ') == masked_name[con_index]:
                score = -1
            else:
                score = 3
                print(response['content'])
 
        except:
            score = 3
            if response is not None:
                print(response['content'])
            
        # save the current winner prediction prompt to the log dir
        prompt_file = debate_name.replace(" ", "_").replace("/", "_").replace("'", "").replace('"', '')
        with open(dir_name+f'/{prompt_file}_winner_predict_prompt.txt', 'w') as f:
            f.write(combined_prompt) 
        
        debate_pred[debate_name] = score

        with open(f'experiments/{file_prefix}_{model_name}_{args.seed}_{args.debate_num}_remove_forfeit_change_names_shuffled-MN.json', 'a') as f:
            json.dump({debate_name: debate_pred[debate_name]}, f)
            f.write('\n') 
        

        num_tokens = response['total_tokens'] if response else 0
        tokens_this_minute += num_tokens

    # get the accuracy, precision, recall, and f1 score
    y_true = []
    y_pred = []
    for name in selected_debate_names:
        if name in debate_pred and debate_pred[name] in [-1, 1]:
            y_pred.append(debate_pred[name])
            y_true.append(int(gt_labels[name]))
        else:
            print(name)
            # print(debate_pred[name])
        

    print("Accuracy: ", accuracy_score(y_true, y_pred))
    print("Precision: ", precision_score(y_true, y_pred, average='binary'))
    print("Recall: ", recall_score(y_true, y_pred, average='binary'))
    print("F1: ", f1_score(y_true, y_pred, average='binary'))
    print("Total number of debate: ", len(y_true))
    
    with open(f'experiments/{file_prefix}_gt_labels_{args.seed}_{args.debate_num}_remove_forfeit.json', 'w') as f:
        json.dump(gt_labels, f)
        