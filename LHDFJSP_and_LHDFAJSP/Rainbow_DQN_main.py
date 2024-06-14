import torch
import numpy as np
import gym
from torch.utils.tensorboard import SummaryWriter
from replay_buffer import *
from rainbow_dqn import DQN
import argparse
from Instance_Generator import Processing_time, A, D, M_num, Op_num, J, O_num, J_num, jobShop_Dict, transportJS, transportMAC
from Job_Shop import Situation
import matplotlib.pyplot as plt


class Runner:
    def __init__(self, args, env_name, number):
        self.args = args
        self.env_name = env_name
        self.number = number
        self.state = np.random.uniform(low=0.00, high=0.01, size=(20,))

        # self.env = gym.make(env_name)  # 修改
        # self.env_evaluate = gym.make(env_name)  # 在评估策略时，需要重建一个环境
        # self.env.action_space.seed(seed)  # 修改
        # self.env_evaluate.seed(seed)
        # self.env_evaluate.action_space.seed(seed)
        # np.random.seed(seed)
        # torch.manual_seed(seed)

        self.args.state_dim = 20  # 已修
        self.args.action_dim = 10  # 已修
        self.args.episode_limit = O_num  # 每episode的最大步数
        print("env={}".format(self.env_name))
        print("state_dim={}".format(self.args.state_dim))
        print("action_dim={}".format(self.args.action_dim))
        print("episode_limit={}".format(self.args.episode_limit))

        if args.use_per and args.use_n_steps:
            self.replay_buffer = N_Steps_Prioritized_ReplayBuffer(args)
        elif args.use_per:
            self.replay_buffer = Prioritized_ReplayBuffer(args)
        elif args.use_n_steps:
            self.replay_buffer = N_Steps_ReplayBuffer(args)
        else:
            self.replay_buffer = ReplayBuffer(args)
        self.agent = DQN(args)

        self.algorithm = 'DQN'
        if args.use_double and args.use_dueling and args.use_noisy and args.use_per and args.use_n_steps:
            self.algorithm = 'Rainbow_' + self.algorithm
        else:
            if args.use_double:
                self.algorithm += '_Double'
            if args.use_dueling:
                self.algorithm += '_Dueling'
            if args.use_noisy:
                self.algorithm += '_Noisy'
            if args.use_per:
                self.algorithm += '_PER'
            if args.use_n_steps:
                self.algorithm += "_N_steps"

        self.writer = SummaryWriter(log_dir='runs/DQN/{}_env_{}_number_{}'.format(self.algorithm, env_name, number))

        self.evaluate_num = 0  # 记录评估次数
        self.evaluate_rewards = []  # 在评估过程中记录奖励
        self.total_steps = 0  # 记录训练过程中的总步数
        if args.use_noisy:  # 如果使用Noisy net，就不需要epsilon贪心策略了
            self.epsilon = 0
        else:
            self.epsilon = self.args.epsilon_init
            self.epsilon_min = self.args.epsilon_min
            self.epsilon_decay = (self.args.epsilon_init - self.args.epsilon_min) / self.args.epsilon_decay_steps

    def reset(self):
        self.state = np.random.uniform(low=0.00, high=0.01, size=(20,))  # 利用均匀随机分布初试化环境的状态
        return np.array(self.state)

    def state_reshape(self, state):
        state = np.array(state)
        shape = state.shape
        # return np.reshape(state, [1, shape[0], 1])
        return np.reshape(state, [1, shape[0]])

    def run(self, J_num, M_num, O_num, J, Processing_time, D, A, Op_num, jobShop_Dict, transportJS, transportMAC):
        x = []  # 计数
        Total_tard = []
        Total_energy = []
        TR = []
        min_energy = 0
        min_tard = 0
        self.evaluate_policy(J_num, M_num, O_num, J, Processing_time, D, A, Op_num, jobShop_Dict, transportJS, transportMAC)
        for episode in range(2000):
            arr = []
            total_energy = 0
            Total_reward = 0
            episode_steps = 0
            x.append(episode + 1)
            print('-----------------------开始第', self.total_steps / O_num + 1, '次训练------------------------------')

            state = self.reset()
            state = np.expand_dims(state, 0)
            done = False
            Sit = Situation(J_num, M_num, O_num, J, Processing_time, D, A, Op_num, jobShop_Dict, transportJS,
                            transportMAC)
            # while not done:  # done 为 FALSE则继续执行
            for i in range(O_num):
                action = self.agent.choose_action(state, epsilon=self.epsilon)
                if action == 0:
                    at_trans = Sit.rule1_2()
                if action == 1:
                    at_trans = Sit.rule1_4()
                if action == 2:
                    at_trans = Sit.rule2_2()
                if action == 3:
                    at_trans = Sit.rule2_4()
                if action == 4:
                    at_trans = Sit.rule2_5()
                if action == 5:
                    at_trans = Sit.rule3_2()
                if action == 6:
                    at_trans = Sit.rule3_4()
                if action == 7:
                    at_trans = Sit.rule5_4()
                if action == 8:
                    at_trans = Sit.rule5_5()
                if action == 9:
                    at_trans = Sit.rule7_4()
                Sit.scheduling(at_trans)

                Job = Sit.Jobs[at_trans[0]]
                start = Job.Start[-1]
                step = len(Job.End) - 1
                row = [int(at_trans[1]) + 1, int(at_trans[0]) + 1, int(step), int(start)]
                arr.append(row)
                print(row)
                print('这是第', i, '道工序>>', '执行action:', action, ' ', '将工件', at_trans[0], '安排到机器',
                      at_trans[1], '开始时间', start, '工序', step)
                time = Processing_time[at_trans[0]][step][at_trans[1]]
                print('加工时间', time)
                if (time < 0): raise Exception('选择了不应该选择的机器')

                next_state = Sit.Features()  # t时刻状态特征
                next_state = np.array(next_state)
                next_state = self.state_reshape(next_state)
                r_t = Sit.Immediate_reward(state[0][6], state[0][5], next_state[0][6], next_state[0][5],
                                           state[0][0],
                                           next_state[0][0], state[0][7], next_state[0][7], state[0][8],
                                           next_state[0][8])
                total_energy += next_state[0][9]
                # total_energy = round(total_energy, 2)
                Job_temp = Sit.Jobs
                current_tardiness = 0
                if i == O_num - 1:
                    for Ji in range(len(Job_temp)):
                        if not Job_temp[Ji].End:
                            current_tardiness += 0
                        else:
                            if max(Job_temp[Ji].End) > D[Ji]:
                                current_tardiness += abs(max(Job_temp[Ji].End) - D[Ji])
                # -----------------------------------------------------------------------------
                if min_tard == 0 and min_energy == 0:
                    r_t += Sit.Cumulative_reward(0, 0, current_tardiness, total_energy)
                else:
                    r_t += Sit.Cumulative_reward(min_tard, min_energy, current_tardiness, total_energy)
                r_t += Sit.mac_based_reward(next_state[0][10], next_state[0][11], next_state[0][12],
                                            next_state[0][13], next_state[0][14], next_state[0][15],
                                            next_state[0][16], next_state[0][17], next_state[0][18],
                                            next_state[0][19])
                if i == O_num - 1:
                    r_t += -(0.5 * current_tardiness * (1 / 80) + 0.5 * total_energy * (1 / 800))

                episode_steps += 1
                self.total_steps += 1

                if not self.args.use_noisy:  # Decay epsilon
                    self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon - self.epsilon_decay > self.epsilon_min else self.epsilon_min

                # 当工序结束或达到max_episode_steps时，done将为true，我们需要区分它们;
                # Terminal的意思是工序结束，没有下一个状态 s';
                # 但是当到达max_episode_steps时，实际上还有下一个状态s'
                if done or episode_steps == self.args.episode_limit:
                    terminal = True
                else:
                    terminal = False

                self.replay_buffer.store_transition(state, action, r_t, next_state, terminal, done)  # 存储转换

                Total_reward += r_t
                state = next_state

                if self.replay_buffer.current_size >= self.args.batch_size:
                    self.agent.learn(self.replay_buffer, self.total_steps)

                if self.total_steps % self.args.evaluate_freq == 0:
                    self.evaluate_policy(J_num, M_num, O_num, J, Processing_time, D, A, Op_num, jobShop_Dict, transportJS, transportMAC)

            total_tardiness = 0
            Job = Sit.Jobs
            End = []
            for Ji in range(len(Job)):
                End.append(max(Job[Ji].End))
                if max(Job[Ji].End) > D[Ji]:
                    total_tardiness += abs(max(Job[Ji].End) - D[Ji])
            # total_tardiness = round(total_tardiness, 2)
            print('<<<<<<<<<-----------------total_tardiness:', total_tardiness, '------------------->>>>>>>>>>')
            Total_tard.append(total_tardiness)
            print('<<<<<<<<<-----------------total_energy:', total_energy, '------------------->>>>>>>>>>')
            Total_energy.append(total_energy)
            print('<<<<<<<<<-----------------reward:', Total_reward, '------------------->>>>>>>>>>')
            TR.append(Total_reward)


            if min_tard == 0 and min_energy == 0:
                min_energy = total_energy
                min_tard = total_tardiness
            elif total_energy < min_energy and total_tardiness < min_tard:
                min_energy = total_energy
                min_tard = total_tardiness
            print(min_tard)
            print(min_energy)
            if episode_steps == self.args.episode_limit:  # 判断是否最后一道工序
                r = np.mean(TR)
                print("episode number: ", episode_steps, ", reward: ", r)
                # break
        plt.plot(x, TR, color='darkred')
        plt.show()
        plt.plot(x, Total_energy, color='darkblue')
        plt.show()
        plt.plot(x, Total_tard, color='y')
        plt.show()
        print("X值", x)
        print("能耗", Total_energy)
        print("延迟时间", Total_tard)
        print("奖励", TR)
        # 保存奖励
        np.save('./data_train/{}_env_{}_number_{}.npy'.format(self.algorithm, self.env_name, self.number), np.array(self.evaluate_rewards))

    def evaluate_policy(self, J_num, M_num, O_num, J, Processing_time, D, A, Op_num, jobShop_Dict, transportJS, transportMAC):
        evaluate_reward = 0
        self.agent.net.eval()
        # Total_tard = []
        # Total_energy = []
        min_energy = 0
        min_tard = 0
        episode_steps = 0
        for _ in range(self.args.evaluate_times):
            state = self.reset()
            state = np.expand_dims(state, 0)
            done = False
            episode_reward = 0
            Sit = Situation(J_num, M_num, O_num, J, Processing_time, D, A, Op_num, jobShop_Dict, transportJS,
                            transportMAC)
            total_energy = 0

            for i in range(O_num):
                action = self.agent.choose_action(state, epsilon=0)
                if action == 0:
                    at_trans = Sit.rule1_2()
                if action == 1:
                    at_trans = Sit.rule1_4()
                if action == 2:
                    at_trans = Sit.rule2_2()
                if action == 3:
                    at_trans = Sit.rule2_4()
                if action == 4:
                    at_trans = Sit.rule2_5()
                if action == 5:
                    at_trans = Sit.rule3_2()
                if action == 6:
                    at_trans = Sit.rule3_4()
                if action == 7:
                    at_trans = Sit.rule3_5()
                if action == 8:
                    at_trans = Sit.rule5_5()
                if action == 9:
                    at_trans = Sit.rule7_4()
                Sit.scheduling(at_trans)
                next_state = Sit.Features()  # t时刻状态特征
                next_state = np.array(next_state)
                next_state = self.state_reshape(next_state)
                r_t = Sit.Immediate_reward(state[0][6], state[0][5], next_state[0][6], next_state[0][5],
                                           state[0][0], next_state[0][0], state[0][7], next_state[0][7],
                                           state[0][8], next_state[0][8])
                total_energy += next_state[0][9]
                # total_energy = round(total_energy, 2)
                Job_temp = Sit.Jobs
                current_tardiness = 0
                if i == O_num - 1:
                    for Ji in range(len(Job_temp)):
                        if not Job_temp[Ji].End:
                            current_tardiness += 0
                        else:
                            if max(Job_temp[Ji].End) > D[Ji]:
                                current_tardiness += abs(max(Job_temp[Ji].End) - D[Ji])
                # -----------------------------------------------------------------------------
                if min_tard == 0 and min_energy == 0:
                    r_t += Sit.Cumulative_reward(0, 0, current_tardiness, total_energy)
                else:
                    r_t += Sit.Cumulative_reward(min_tard, min_energy, current_tardiness, total_energy)
                r_t += Sit.mac_based_reward(next_state[0][10], next_state[0][11], next_state[0][12],
                                            next_state[0][13], next_state[0][14], next_state[0][15],
                                            next_state[0][16], next_state[0][17], next_state[0][18],
                                            next_state[0][19])
                if i == O_num - 1:
                    r_t += -(0.5 * current_tardiness * (1 / 80) + 0.5 * total_energy * (1 / 800))
                episode_reward += r_t
                state = next_state
                episode_steps += 1
                if episode_steps == self.args.episode_limit:  # 判断是否最后一道工序
                    done = True
            total_tardiness = 0
            Job = Sit.Jobs
            End = []
            for Ji in range(len(Job)):
                End.append(max(Job[Ji].End))
                if max(Job[Ji].End) > D[Ji]:
                    total_tardiness += abs(max(Job[Ji].End) - D[Ji])
            # total_tardiness = round(total_tardiness, 2)
            if min_tard == 0 and min_energy == 0:
                min_energy = total_energy
                min_tard = total_tardiness
            elif total_energy < min_energy and total_tardiness < min_tard:
                min_energy = total_energy
                min_tard = total_tardiness
            evaluate_reward += episode_reward
        self.agent.net.train()
        evaluate_reward /= self.args.evaluate_times
        self.evaluate_rewards.append(evaluate_reward)
        print("total_steps:{} \t evaluate_reward:{} \t epsilon：{}".format(self.total_steps, evaluate_reward, self.epsilon))
        self.writer.add_scalar('step_rewards_{}'.format(self.env_name), evaluate_reward, global_step=self.total_steps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("DQN超参数设置")
    parser.add_argument("--max_train_steps", type=int, default=int(4e5), help=" 最大训练步数")
    parser.add_argument("--evaluate_freq", type=float, default=1e3, help="每个“evaluate_freq”步骤评估策略")
    parser.add_argument("--evaluate_times", type=float, default=3, help="评估时间")

    parser.add_argument("--buffer_capacity", type=int, default=int(1e5), help="最大replay buffer容量 ")  # 1e5
    parser.add_argument("--batch_size", type=int, default=256, help="批处理大小")  # 256
    parser.add_argument("--hidden_dim", type=int, default=256, help="神经网络隐藏层中的神经元数量")
    parser.add_argument("--lr", type=float, default=1e-4, help="actor 学习率")  # 1e-4
    parser.add_argument("--gamma", type=float, default=0.99, help="折扣因子")
    parser.add_argument("--epsilon_init", type=float, default=0.5, help="初始的epsilon")
    parser.add_argument("--epsilon_min", type=float, default=0.1, help="最小的epsilon")
    parser.add_argument("--epsilon_decay_steps", type=int, default=int(1e5), help="衰变到最小值需要多少步")
    parser.add_argument("--tau", type=float, default=0.005, help="软更新目标网络")
    parser.add_argument("--use_soft_update", type=bool, default=True, help="是否使用 soft update")
    parser.add_argument("--target_update_freq", type=int, default=200, help="目标网络的更新频率(硬更新)")
    parser.add_argument("--n_steps", type=int, default=5, help="n_steps")
    parser.add_argument("--alpha", type=float, default=0.6, help="PER parameter")
    parser.add_argument("--beta_init", type=float, default=0.4, help="PER中重要的抽样参数")  # 0.4
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="学习率衰减")
    parser.add_argument("--grad_clip", type=float, default=10.0, help="Gradient clip")

    parser.add_argument("--use_double", type=bool, default=True, help="是否使用 double Q-learning")
    parser.add_argument("--use_dueling", type=bool, default=True, help="是否使用 dueling network")
    parser.add_argument("--use_noisy", type=bool, default=True, help="是否使用 noisy network")
    parser.add_argument("--use_per", type=bool, default=True, help="是否使用 PER")
    parser.add_argument("--use_n_steps", type=bool, default=True, help="是否使用 n_steps Q-learning")

    args = parser.parse_args()
    print(args)
    env_names = ['Low-carbon Distributed Flexible job shop scheduling', 'Low-carbon Distributed Flexible Assembly job shop scheduling']
    env_index = 1
    for seed in [0, 10, 100]:
        runner = Runner(args=args, env_name=env_names[env_index], number=1)
        runner.run(J_num, M_num, O_num, J, Processing_time, D, A, Op_num, jobShop_Dict, transportJS, transportMAC)
