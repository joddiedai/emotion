import numpy as np
import pandas as pd
import pickle

N_STATES = 161
ACTIONS = ['av', 'tv', 'at', 'avt', 'a', 'v', 't']     # available actions
EPSILON = 0.9   # greedy police
ALPHA = 0.1     # learning rate
GAMMA = 0.9    # discount factor
MAX_EPISODES = 161   # maximum episodes

def build_q_table(): # initialize Q-table
    data = {'to_delete': np.zeros(N_STATES)}
    df = pd.DataFrame(data)
    pickle_list = ['at', 'av', 'tv', 'avt', 'a', 'v', 't']
    for a in pickle_list:
        f = open('features_{0}_1b_160e.pickle'.format(a),'rb')
        pk = pickle.load(f)
        f.close()
        pk = np.array(pk)
        acc_list = pk[:, 1].tolist()
        acc_list_ = []
        for b in acc_list:
            acc_list_.append(b)
        acc_list_ = np.array(acc_list_)
        df.insert(0, a, acc_list_)
    del df['to_delete']
    return df

def choose_action(state, q_table):
    # This is how to choose an action
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()):  # act non-greedy or state-action have no value
        action_name = np.random.choice(ACTIONS)
    else:   # act greedy
        action_name = state_actions.idxmax()    # replace argmax to idxmax as argmax means a different function in newer version of pandas
    return action_name

def rl():
    q_table = build_q_table()
    for episode in range(MAX_EPISODES):
        path = []
        path_number = []
        step_counter = 0
        S = 0
        is_terminated = False
        # update_env(S, episode, step_counter)
        while S < N_STATES-1:

            A = choose_action(S, q_table)
            if A == 'av':
                path_number.append(0)
            if A == 'tv':
                path_number.append(1)
            if A == 'at':
                path_number.append(2)
            if A == 'avt':
                path_number.append(3)
            if A == 'a':
                path_number.append(4)
            if A == 'v':
                path_number.append(5)
            if A == 't':
                path_number.append(6)
            path.append(A)
            #S_, R = get_env_feedback(S, A)  # take action & get next state and reward
            S_, R = S+1, 0
            q_predict = q_table.loc[S, A]
            if S_ != 'terminal':
                q_target = R + GAMMA * q_table.iloc[S_, :].max()   # next state is not terminal
            else:
                q_target = R     # next state is terminal
                is_terminated = True    # terminate this episode

            q_table.loc[S, A] += ALPHA * (q_target - q_predict)  # update
            S = S_  # move to next state

            step_counter += 1
    print(path)
    print(path_number)
    return q_table, path, path_number

if __name__ == "__main__":
    list = rl()[2]
    av = 0
    at = 0
    avt = 0
    tv = 0
    a = 0
    v = 0
    t = 0
    for i in list:
        if i == 0:
            av += 1
        if i == 2:
            at += 1
        if i == 3:
            avt += 1
        if i == 1:
            tv += 1
        if i == 4:
            a += 1
        if i == 5:
            v += 1
        if i == 6:
            t += 1
    print('av,',av, 'at,',at, 'tv,',tv, 'avt,',avt, 'a,', a, 'v,', v, 't,', t)