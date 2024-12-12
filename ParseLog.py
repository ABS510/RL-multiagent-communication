import io
import json
import os
import re
import argparse

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

agents = ['first_0', 'second_0', 'third_0', 'fourth_0']
behaviors = ['followed', 'not_followed', 'no_intention']
summary_cols = ['intention followed',
                'intention not followed',
                'no intentions']

def read_log(log_fname, out_dir, msg=None):
    msg_suffix = ""
    if msg is not None:
        msg_suffix = f"({msg})"

    training_game = 0
    eval_game = -1

    train_df_list = []
    eval_df_list = []
    eval_summary_list = []

    with open(log_fname, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            
            game_start_match = re.match(r'.* - VolleyballPongEnv - INFO - Game (\d+), epsilon: (\d+)', line)
            if game_start_match:
                training_game = int(game_start_match.group(1))
                
                while len(train_df_list) < training_game + 1:
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
                eval_summary_list.append({
                    'Training_Game': training_game,
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

            # >>> res = re.match(r'.* - VolleyballPongEnv - INFO - Evaluation: Agent (\w+_0) Intention Metrics: (.*)', line)
            eval_intention_match = re.match(r'.* - VolleyballPongEnv - INFO - Evaluation: Agent (\w+_0) Intention Metrics: (.*)', line)
            if eval_intention_match:
                agent_name = eval_intention_match.group(1)
                intention_dict_str = eval_intention_match.group(2).replace("'", '"') # use double quotes instead of single 
                intention_dict = json.loads(intention_dict_str)
                for behavior in intention_dict:
                    col_name = f"{agent_name}_{behavior}"
                    eval_df_list[-1][col_name] = intention_dict[behavior]
                continue

            eval_mean_match = re.match(r'.* - VolleyballPongEnv - INFO - Evaluation: Agent (\w+_0) mean (.+): (.*)', line)
            if eval_mean_match:
                agent_name = eval_mean_match.group(1)
                item_name = eval_mean_match.group(2).replace("-", " ")
                item_mean = eval_mean_match.group(3)
                eval_summary_list[-1][f"{agent_name}_{item_name}_mean"] = float(item_mean)
                continue

            
            eval_std_match = re.match(r'.* - VolleyballPongEnv - INFO - Evaluation: Agent (\w+_0) (.+) std: (.*)', line)
            if eval_std_match:
                agent_name = eval_std_match.group(1)
                item_name = eval_std_match.group(2).replace("-", " ")
                item_std = eval_std_match.group(3)
                eval_summary_list[-1][f"{agent_name}_{item_name}_std"] = float(item_std)
                continue
            
                
    train_df = pd.DataFrame(train_df_list).dropna()

    if not train_df.empty: 
        train_df.to_csv(os.path.join(out_dir, 'train_log.csv'), index_label='Game')
        for suffix in ['_reward', '_loss', '_penalty']:
            if suffix == '_penalty' and 'first_0_penalty' not in train_df:
                continue
            train_item_df = train_df[[a+suffix for a in agents]].rename(columns={
                (a+suffix) : str(i+1) for i, a in enumerate(agents)
            })

            train_plot = sns.lineplot(data=train_item_df) 
            if suffix == '_loss':
                plt.yscale('log')
            train_plot.set_xlabel("Training Game Number")

            suffix_title = suffix.replace("_", "").title()
            train_plot.set_ylabel(suffix_title)
            train_plot.set_title(f"Training {suffix_title} {msg_suffix}")
            train_plot.figure.savefig(os.path.join(out_dir, f"train{suffix}.png"))
            train_plot.figure.clear()

            if suffix == '_reward':
                train_win_df = pd.DataFrame({
                    'win_reward': train_item_df.max(axis=1),
                    'lose_reward': train_item_df.min(axis=1)
                })
                train_win_plot = sns.lineplot(data=train_win_df) 
                train_win_plot.set_xlabel("Training Game Number")
                train_win_plot.set_ylabel("Reward")
                train_win_plot.figure.savefig(os.path.join(out_dir, f"train_win_reward.png"))
                train_win_plot.figure.clear()

    eval_df = pd.DataFrame(eval_df_list).dropna()

    if not eval_df.empty:
        eval_df.to_csv(os.path.join(out_dir, 'eval_log.csv'), index=False)
        for suffix in ['_reward', '_score']:
            eval_item_df = eval_df[[a+suffix for a in agents]]
            eval_plot = sns.lineplot(data=eval_item_df) 
            eval_plot.set_xlabel("Evaluation Period")
            suffix_title = suffix.replace("_", "").title()
            eval_plot.set_ylabel(suffix_title)
            eval_plot.set_title(f"Evaluation {suffix_title} {msg_suffix}")
            eval_plot.figure.savefig(os.path.join(out_dir, f"eval{suffix}.png"))
            eval_plot.figure.clear()

            if suffix == '_score':
                eval_win_df = pd.DataFrame({
                    'win_reward': eval_item_df.max(axis=1),
                    'lose_reward': eval_item_df.min(axis=1)
                })
                eval_win_plot = sns.lineplot(data=eval_win_df) 
                eval_win_plot.set_xlabel("Evaluation Period")
                eval_win_plot.set_ylabel("Reward")
                eval_win_plot.figure.savefig(os.path.join(out_dir, f"eval_win{suffix}.png"))
                eval_win_plot.figure.clear()

        try:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            for i, behavior in enumerate(behaviors):
                eval_behavior_df = eval_df[[f"{a}_{behavior}" for a in agents]]
                eval_behavior_df = eval_behavior_df.rename(columns={
                    f"{a}_{behavior}" : a
                    for a in agents
                })
                sns.lineplot(data=eval_behavior_df, ax=axes[i])
                axes[i].set_xlabel("Evaluation Period")
                axes[i].set_ylabel("Count")
                axes[i].set_title(f"{behavior} {msg_suffix}")
            fig.savefig(os.path.join(out_dir, "eval_intentions.png"))
            fig.clear()
        except:
            print("No intention information found!")
            
    try:
        eval_summary_df = pd.DataFrame(eval_summary_list).dropna()
        if not eval_summary_df.empty:
            eval_summary_df.to_csv(os.path.join(out_dir, 'eval_summary_log.csv'), index=False)
            x = np.arange(len(eval_summary_df))

            # # hack
            # if len(eval_summary_df) < 2:
            #     eval_summary_df = pd.concat([eval_summary_df]*5,ignore_index=True)

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            for i, c in enumerate(['reward', 'score']):
                for a in agents:
                    # sns.lineplot(eval_summary_df[f"{a}_{c}_mean"], ax=axes[i])
                    y = eval_summary_df[f"{a}_{c}_mean"].to_numpy()
                    yerr = eval_summary_df[f"{a}_{c}_std"].to_numpy()
                    _, _, bars = axes[i].errorbar(x, y, yerr=yerr, label=a)
                    for bar in bars:
                        bar.set_alpha(0.3) 
                axes[i].set_title(f"{c} {msg_suffix}")
                axes[i].set_xlabel("Evaluation Period")
                axes[i].set_ylabel("Score")
                axes[i].legend()
            fig.savefig(os.path.join(out_dir, "eval_summary_scores.png"))
            fig.clear()
            
            fig, axes = plt.subplots(1, 5, figsize=(25, 5))
            for i, a in enumerate(agents):
                means = {
                    c: eval_summary_df[f"{a}_{c}_mean"].to_numpy()
                    for c in summary_cols
                }
                # print([v for v in means.values()])
                # print(np.vstack([v for v in means.values()]))
                total_actions_cnt = np.sum(np.stack([v for v in means.values()]), axis=0)
                # total_actions_cnt = 1
                stack_items = [
                    v / total_actions_cnt 
                    for v in means.values()
                ]
                axes[i].stackplot(x, *stack_items, labels=means.keys())
                axes[i].set_xlabel("Evaluation Period")
                axes[i].set_ylabel("Relative Frequency")
                axes[i].set_title(f"{a} Behavior {msg_suffix}")
                axes[i].legend()
            
            overall_means = {
                c: eval_summary_df[[f"{a}_{c}_mean" for a in agents]].mean(axis=1).to_numpy()
                for c in summary_cols
            }
            total_actions_cnt_overall = np.sum(np.stack([v for v in overall_means.values()]), axis=0)
            # total_actions_cnt_overall = 1
            total_stack_items = [
                v / total_actions_cnt_overall 
                for v in overall_means.values()
            ]
            axes[-1].stackplot(x, *total_stack_items, labels=overall_means.keys())
            axes[-1].set_xlabel("Evaluation Period")
            axes[-1].set_ylabel("Relative Frequency")
            
            # axes[-1].stackplot(x, overall_means.values(), labels=overall_means.keys())
            axes[-1].set_title(f"Average Behavior {msg_suffix}")
            axes[-1].legend()

            fig.savefig(os.path.join(out_dir, "eval_summary_intentions.png"))
    except:
        print("No mean/std lines found!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Loading a log file")
    parser.add_argument(
        "-i", "--log", type=str, required=True, help="Path to the log .log file",
    )
    parser.add_argument(
        "-o", "--out", type=str, required=True, help="Path to the .csv output folder"
    )
    parser.add_argument(
        "-m", "--message", type=str, required=False, help="Optional title label"
    )

    args = parser.parse_args()
    log_fname = args.log
    out_dir = args.out
    msg = args.message

    if not os.path.exists(log_fname):
        raise FileNotFoundError

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    read_log(log_fname, out_dir, msg)





    
