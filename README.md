# Rainbow DQN Based Job Shop Scheduling Method (low-carbon heterogeneous flexible job shop)
Rainbow DQN is implemented using Pytorch, including Double DQN, dueiling DQN, noise network, PER and multi-step learning <br />

# Introduction
First, the corresponding scheduling environment of LHDFJSP and LHDFAJSP is established, which satisfies the constraints of the mathematical model. Secondly, the architecture, state space, action space and reward function of Rainbow DQN algorithm are established. For LHDFAJSP, the architecture, state space, action space and reward function of DDS-Rainbow DQN algorithm are established.

## Components
python==3.7.9<br />
numpy==1.19.4<br />
pytorch==1.5.0<br />
tensorboard==0.6.0<br />
gym==0.21.0<br />

## Use
First deploy the project, configure the environment, and run Rainbow_DQN_main.py directly in your personal IDE<br />

### Trainning
You can set the 'env_index' in the code to change the environments.<br />
env_index=0 represent 'Low-carbon Distributed Flexible job shop scheduling'<br />
env_index=1 represent 'Low-carbon Distributed Flexible Assembly job shop scheduling'<br />

### How to see the training results?
You can use the tensorboard to visualize the training curves, which are saved in the file 'runs'.<br />
Or you can run the program to visualize the training curves after the end of program.<br />
The rewards data are saved as numpy in the file 'data_train'.<br />
The training curves are shown below.<br />



