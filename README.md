[![GitHub license](https://img.shields.io/github/license/mail-ecnu/PICO)](https://github.com/mail-ecnu/PICO/blob/master/LICENSE)
![Read the Docs](https://img.shields.io/readthedocs/pico)
[![GitHub issues](https://img.shields.io/github/issues/mail-ecnu/PICO)](https://github.com/mail-ecnu/PICO/issues)
[![GitHub forks](https://img.shields.io/github/forks/mail-ecnu/PICO)](https://github.com/mail-ecnu/PICO/network)
[![GitHub stars](https://img.shields.io/github/stars/mail-ecnu/PICO)](https://github.com/mail-ecnu/PICO/stargazers)

**PICO** is an algorithm for exploiting Reinforcement Learning (RL) on Multi-agent Path Finding tasks.
It is developed by the [Multi-Agent Artificial Intelligence Lab (MAIL)](https://mail-ecnu.cn) in East China Normal University and the AI Research Institute in [Geekplus Technology Co., Ltd](https://www.geekplus.com/).
PICO is constructed based on the framework of [*PRIMAL:Pathfinding via Reinforcement and Imitation Multi-Agent Learning*](https://github.com/gsartoretti/PRIMAL) and focuses more on the collision avoidance rather than manual post-processing when collision occurs.
Exploiting the design of decentralized communication and implicit priority in these secenarios benifits better path finding.
To emphasis, more details about PICO can be found in our paper [*Multi-Agent Path Finding with Prioritized Communication Learning*](https://arxiv.org/abs/2202.03634), which is accepted by **ICRA 2022**.

## Distributed Assembly
Reinforcement learning code to train multiple agents to
collaboratively plan their paths in a 2D grid world.

### Key Components of PICO

- pico_training.py: Multi-agent training code. Training
runs on GPU by default, change line "with tf.device("/gpu:0"):"
to "with tf.device("/cpu:0"):" to train on CPU (much slower).Researchers can also flexibly customized their configuration in this file.
- mapf_gym.py: Multi-agent path planning gym environment,
in which agents learn collective path planning.
- pico_testing.py: Code to run systematic validation tests
of PICO, pulled from the saved_environments folder as .npy
files and output results in a given folder (by default: test_result).

## Installation 

```
git clone https://github.com/mail-ecnu/PICO.git
cd PICO
conda env create -f conda_env.yml
conda activate PICO-dev
```
## Before compilation: compile cpp_mstar code

- cd into the od_mstar3 folder.
- python3 setup.py build_ext (may need --inplace as extra argument).
- copy so object from build/lib.*/ at the root of the od_mstar3 folder.
- Check by going back to the root of the git folder,
running python3 and "import cpp_mstar"

## Quick Examples
pico_training.py:
```yaml
episode_count          = 0
MAX_EPISODE            = 20
EPISODE_START          = episode_count
gamma                  = .95 # discount rate for advantage estimation and reward discounting
#moved network parameters to ACNet.py
EXPERIENCE_BUFFER_SIZE = 128
GRID_SIZE              = 11 #the size of the FOV grid to apply to each agent
ENVIRONMENT_SIZE       = (10,20)#(10,70) the total size of the environment (length of one side)
OBSTACLE_DENSITY       = (0,0.3) #(0,0.5) range of densities
DIAG_MVMT              = False # Diagonal movements allowed?
a_size                 = 5 + int(DIAG_MVMT)*4
SUMMARY_WINDOW         = 10
NUM_META_AGENTS        = 3
NUM_THREADS            = 8 #int(multiprocessing.cpu_count() / (2 * NUM_META_AGENTS))
# max_episode_length     = 256 * (NUM_THREADS//8)
max_episode_length     = 256
NUM_BUFFERS            = 1 # NO EXPERIENCE REPLAY int(NUM_THREADS / 2)
EPISODE_SAMPLES        = EXPERIENCE_BUFFER_SIZE # 64
LR_Q                   = 2.e-5
ADAPT_LR               = True
ADAPT_COEFF            = 5.e-5 #the coefficient A in LR_Q/sqrt(A*steps+1) for calculating LR
load_model             = False
RESET_TRAINER          = False
gifs_path              = 'gifs'
from datetime import datetime
TIMESTAMP = "{0:%Y-%m-%dT%H-%M/}".format(datetime.now())

GLOBAL_NET_SCOPE       = 'global'

#Imitation options
PRIMING_LENGTH         = 2500    #0 number of episodes at the beginning to train only on demonstrations
DEMONSTRATION_PROB     = 0.5
```

Then 
```sh
python pico_training.py
```

### Custom testing

Edit pico_testing.py according to the training setting.
By default, the model is loaded from the *model* folder.

Then 
```sh
python pico_testing.py
```

### Requirements
- Python 3.4
- Cython 0.28.4
- OpenAI Gym 0.9.4
- Tensorflow 1.3.1
- Numpy 1.13.3
- matplotlib
- imageio (for GIFs creation)
- tk
- networkx (if using od_mstar.py and not the C++ version)


### Citing our work
If you use this repo in your work, please consider citing the corresponding paper (first two authors contributed equally):

```bibtex
@InProceedings{lichen2022mapf,
  title =    {Multi-Agent Path Finding with Prioritized Communication Learning},
  author =   {Wenhao, Li* and Hongjun, Chen* and Bo, Jin and Wenzhe, Tan and Hongyuan, Zha and Xiangfeng, Wang},
  booktitle =    {ICRA},
  year =     {2022},
  pdf =      {https://arxiv.org/pdf/2202.03634},
  url =      {https://arxiv.org/abs/2202.03634},
}
```


## License
Licensed under the [MIT](./LICENSE) License.
