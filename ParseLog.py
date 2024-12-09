import io
import os
import re
import argparse

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

agents = ['first_0', 'second_0', 'third_0', 'fourth_0']

def read_log(log_fname, out_dir):
    training_game = 0
    eval_game = -1

    train_df_list = []
    eval_df_list = []
    with open(log_fname, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            
            game_start_match = re.match(r'.* - VolleyballPongEnv - INFO - Game (\d+)', line)
            if game_start_match:
                training_game = int(game_start_match.group(1))

                train_df_list.append({})
                continue
            
            # train log
            # newer logs
            train_reward_match = re.match(r'.* - VolleyballPongEnv - INFO - Agent (\w+_0) accumulated reward: (.+), accumulated penalty: (.+)', line)
            if train_reward_match:
                agent_name = train_reward_match.group(1)
                reward = train_reward_match.group(2)
                penalty = train_reward_match.group(3)
                train_df_list[training_game][agent_name + '_reward'] = float(reward)
                train_df_list[training_game][agent_name + '_penalty'] = float(penalty)

                continue

            # older logs
            train_reward_match = re.match(r'.* - VolleyballPongEnv - INFO - Agent (\w+_0) accumulated reward: (.+)', line)
            if train_reward_match:
                agent_name = train_reward_match.group(1)
                reward = train_reward_match.group(2)
                col_name = agent_name + '_reward'

                train_df_list[training_game][col_name] = float(reward)
                continue

            train_loss_match = re.match(r'.* - VolleyballPongEnv - INFO - Agent (\w+_0) average loss: (.+)', line)
            if train_loss_match:
                agent_name = train_loss_match.group(1)
                loss = train_loss_match.group(2)
                col_name = agent_name + '_loss'

                train_df_list[training_game][col_name] = float(loss)
                continue
            
            # eval log
            eval_start_match = re.match(r'.* - VolleyballPongEnv - INFO - Evaluation Game (\d+)', line)
            if eval_start_match:
                eval_game = int(eval_start_match.group(1))
                eval_df_list.append({
                    'Training_Game': training_game,
                    'Evaluation_Game': eval_game
                })
                continue
            
            eval_reward_match = re.match(r'.* - VolleyballPongEnv - INFO - Evaluation: Agent (\w+_0) accumulated reward: (.+)', line)
            if eval_reward_match:
                agent_name = eval_reward_match.group(1)
                reward = eval_reward_match.group(2)
                col_name = agent_name + '_reward'

                eval_df_list[-1][col_name] = float(reward)
                continue

            eval_score_match = re.match(r'.* - VolleyballPongEnv - INFO - Evaluation: Agent (\w+_0) accumulated score: (.+)', line)
            if eval_score_match:
                agent_name = eval_score_match.group(1)
                score = eval_score_match.group(2)
                col_name = agent_name + '_score'

                eval_df_list[-1][col_name] = float(score)
                continue


    train_df = pd.DataFrame(train_df_list).dropna()
    train_df.to_csv(os.path.join(out_dir, 'train_log.csv'), index_label='Game')

    for suffix in ['_reward', '_loss', '_penalty']:
        if suffix == '_penalty' and 'first_0_penalty' not in train_df:
            continue
        train_item_df = train_df[[a+suffix for a in agents]]

        train_plot = sns.lineplot(data=train_item_df) 
        if suffix == '_loss':
            plt.yscale('log')
        train_plot.figure.savefig(os.path.join(out_dir, f"train{suffix}.png"))
        train_plot.figure.clear()

        if suffix == '_reward':
            train_win_df = pd.DataFrame({
                'win_reward': train_item_df.max(axis=1),
                'lose_reward': train_item_df.min(axis=1)
            })
            train_win_plot = sns.lineplot(data=train_win_df) 
            train_win_plot.figure.savefig(os.path.join(out_dir, f"train_win_reward.png"))
            train_win_plot.figure.clear()

    eval_df = pd.DataFrame(eval_df_list).dropna()
    eval_df.to_csv(os.path.join(out_dir, 'eval_log.csv'), index=False)

    for suffix in ['_reward', '_score']:
        eval_item_df = eval_df[[a+suffix for a in agents]]
        eval_plot = sns.lineplot(data=eval_item_df) 
        eval_plot.figure.savefig(os.path.join(out_dir, f"eval{suffix}.png"))
        eval_plot.figure.clear()

        if suffix == '_score':
            eval_win_df = pd.DataFrame({
                'win_reward': eval_item_df.max(axis=1),
                'lose_reward': eval_item_df.min(axis=1)
            })
            eval_win_plot = sns.lineplot(data=eval_win_df) 
            eval_win_plot.figure.savefig(os.path.join(out_dir, f"eval_win{suffix}.png"))
            eval_win_plot.figure.clear()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Loading a log file")
    parser.add_argument(
        "-i", "--log", type=str, required=True, help="Path to the log .log file",
    )
    parser.add_argument(
        "-o", "--out", type=str, required=True, help="Path to the .csv output folder"
    )

    args = parser.parse_args()
    log_fname = args.log
    out_dir = args.out

    if not os.path.exists(log_fname):
        raise FileNotFoundError

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    read_log(log_fname, out_dir)





    
