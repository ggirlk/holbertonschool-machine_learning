# ![](holberton-logo.png) 0x01. Deep Q-learning

<p align="center"><img src="atari.gif" width="60%" height="50%"></p>

# 🧑🏻‍💻 Intro

> Python script that utilizes keras, keras-rl, and gym to train an agent that can play Atari’s Breakout

# 🧠 Learning Objectives
> - What is Deep Q-learning?
> - What is the policy network?
> - What is replay memory?
> - What is the target network?
> - Why must we utilize two separate networks during training?
> - What is keras-rl? How do you use it?

# 📕 Dependencies

> <b> ⛔️ If you have keras-rl2 already installed delete it before you start</b>.

```
pip install --user keras-rl==0.4.2
pip install --user tensorflow==1.14.0
pip install --user keras==2.2.4
pip install --user Pillow==8.0.1
pip install --user h5py==2.10.0
pip install --user atari-py==0.2.6
pip install --user gym==0.18.3
pip install --user gym[atari]

wget http://www.atarimania.com/roms/Roms.rar
unrar e -ad Roms.rar 
python -m atari_py.import_roms Roms
```

# 🕹 How it works

## 🧗‍♀️ Training
The model is already trained and the weights have been saved in <em>policy.h5f</em> file but if you want to redo it again just type

```
python train.py
```
> ⛔️ It takes seviral time
## ⛹️‍♀️ Playing

```
python play.py
```

# 🔗 Resources

- [Deep Q-Learning - Combining Neural Networks and Reinforcement Learning](https://www.youtube.com/watch?v=wrBUkpiRvCA&list=PLZbbT5o_s2xoWNVdDudn51XM8lOuZ_Njv&index=11)
- [Replay Memory Explained - Experience for Deep Q-Network Training](https://www.youtube.com/watch?v=Bcuj2fTH4_4&list=PLZbbT5o_s2xoWNVdDudn51XM8lOuZ_Njv&index=12)
- [Training a Deep Q-Network - Reinforcement Learning](https://www.youtube.com/watch?v=0bt0SjbS3xc&list=PLZbbT5o_s2xoWNVdDudn51XM8lOuZ_Njv&index=13)
- [Training a Deep Q-Network with Fixed Q-targets - Reinforcement Learning](https://www.youtube.com/watch?v=xVkPh9E9GfE&list=PLZbbT5o_s2xoWNVdDudn51XM8lOuZ_Njv&index=14)
- [keras-rl](https://github.com/keras-rl/keras-rl)
    - [rl.policy](https://github.com/keras-rl/keras-rl/blob/master/rl/policy.py)
    - [rl.memory](https://github.com/keras-rl/keras-rl/blob/master/rl/memory.py)
    - [rl.agents.dqn](https://github.com/keras-rl/keras-rl/blob/master/rl/agents/dqn.py)
- [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602.pdf)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/playing-atari-with-deep-reinforcement/atari-games-on-atari-2600-breakout)](https://paperswithcode.com/sota/atari-games-on-atari-2600-breakout?p=playing-atari-with-deep-reinforcement)
<hr>

By [Khouloud](https://www.linkedin.com/in/khouloud-alkhammassi-3a9078129) Software engineer at [HolbertonSchool®️](https://www.holbertonschool.com)