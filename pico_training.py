#!/usr/bin/env python
# coding: utf-8

# 
# While training is taking place, statistics on agent performance are available from Tensorboard. To launch it use:
# 
# `tensorboard --logdir train_log`

# In[ ]:


#this should be the thing, right?
from __future__ import division

import gym
import numpy as np
import random
# import tensorflow as tf
# import tensorflow.contrib.layers as layers
import matplotlib.pyplot as plt
from od_mstar3 import cpp_mstar
from od_mstar3 import od_mstar
from od_mstar3.col_set_addition import OutOfTimeError,NoSolutionError
import threading
import time
import scipy.signal as signal
import os
import GroupLock
import multiprocessing
# get_ipython().run_line_magic('matplotlib', 'inline')
import mapf_gym as mapf_gym
import pickle
import imageio
from ACNetComm_old import ACNet

from tensorflow.python.client import device_lib


import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

import pdb
 


dev_list = device_lib.list_local_devices()
print(dev_list)
assert len(dev_list) > 1

# import ipdb

# ### Helper Functions

# In[ ]:


def make_gif(images, fname, duration=2, true_image=False,salience=False,salIMGS=None):
    imageio.mimwrite(fname,images,subrectangles=True)
    print("wrote gif")

# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.
def update_target_graph(from_scope,to_scope):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder

def discount(x, gamma):
    return signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

def good_discount(x, gamma):
    return discount(x,gamma)


# ## Worker Agent

# In[ ]:

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed) 
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)

class Worker:
    def __init__(self, game, metaAgentID, workerID, a_size, groupLock, max_episode):
        self.workerID = workerID
        self.env = game
        self.metaAgentID = metaAgentID
        self.name = "worker_"+str(workerID)
        self.agentID = ((workerID-1) % num_workers) + 1 
        self.groupLock = groupLock
        self.max_episode = max_episode

        self.nextGIF = episode_count # For GIFs output
        #Create the local copy of the network and the tensorflow op to copy global parameters to local network
        self.local_AC = ACNet(self.name,a_size,trainer,True,GRID_SIZE,GLOBAL_NET_SCOPE)
        # master_network = ACNet(GLOBAL_NET_SCOPE,a_size,None,False,GRID_SIZE,GLOBAL_NET_SCOPE) # Generate global network
        self.pull_global = update_target_graph(GLOBAL_NET_SCOPE, self.name)

    def synchronize(self):
        #handy thing for keeping track of which to release and acquire
        if(not hasattr(self,"lock_bool")):
            self.lock_bool=False
        self.groupLock.release(int(self.lock_bool),self.name)
        self.groupLock.acquire(int(not self.lock_bool),self.name)
        self.lock_bool=not self.lock_bool
        
    def train(self, rollout, sess, gamma, bootstrap_value, rnn_state0, imitation=False):
        global episode_count
        if imitation:
            rollout=np.array(rollout)
            #we calculate the loss differently for imitation
            #if imitation=True the rollout is assumed to have different dimensions:
            #[o[0],o[1],optimal_actions, target_priority(if True)]
            feed_dict={global_step:episode_count,
                       self.local_AC.inputs:np.stack(rollout[:,0]),
                       self.local_AC.goal_pos:np.stack(rollout[:,1]),
                       self.local_AC.optimal_actions:np.stack(rollout[:,2]),
                       self.local_AC.target_priority:np.stack(rollout[:,3]),
                       self.local_AC.message:np.stack(rollout[:,4]),
                       self.local_AC.state_in[0]:rnn_state0[0],
                       self.local_AC.state_in[1]:rnn_state0[1]
                      }
            _,bc_l,pi_l,i_l,_=sess.run([self.local_AC.policy,self.local_AC.behavior_cloning_loss,self.local_AC.priority_loss,self.local_AC.imitation_loss,
                              self.local_AC.apply_imitation_grads],
                             feed_dict=feed_dict)
            return bc_l,pi_l,i_l
        # episode_buffer.append([s[0],a,r,s1,d,v[0,0],train_valid,pred_on_goal,int(on_goal),pred_blocking,int(blocking),message,s[1],train_val])
        rollout = np.array(rollout)
        observations = rollout[:,0]
        goals=rollout[:,-2]
        actions = rollout[:,1]
        rewards = rollout[:,2]
        values = rollout[:,5]
        valids = rollout[:,6]
        blockings = rollout[:,10]
        message = rollout[:,11]
        on_goals=rollout[:,8]
        train_value = rollout[:,-1]

        # Here we take the rewards and values from the rollout, and use them to 
        # generate the advantage and discounted returns. (With bootstrapping)
        # The advantage function uses "Generalized Advantage Estimation"
        self.rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
        discounted_rewards = discount(self.rewards_plus,gamma)[:-1]
        self.value_plus = np.asarray(values.tolist() + [bootstrap_value])
        advantages = rewards + gamma * self.value_plus[1:] - self.value_plus[:-1]
        advantages = good_discount(advantages,gamma)

        num_samples = min(EPISODE_SAMPLES,len(advantages))
        sampleInd = np.sort(np.random.choice(advantages.shape[0], size=(num_samples,), replace=False))

        # Update the global network using gradients from loss
        # Generate network statistics to periodically save
        feed_dict = {
            global_step:episode_count,
            self.local_AC.target_v:np.stack(discounted_rewards),
            self.local_AC.inputs:np.stack(observations),
            self.local_AC.goal_pos:np.stack(goals),
            self.local_AC.message:np.stack(message),
            self.local_AC.actions:actions,
            self.local_AC.train_valid:np.stack(valids),
            self.local_AC.advantages:advantages,
            self.local_AC.train_value:train_value,
            self.local_AC.target_blockings:blockings,
            self.local_AC.target_on_goals:on_goals,
            self.local_AC.state_in[0]:rnn_state0[0],
            self.local_AC.state_in[1]:rnn_state0[1]
        }

        v_l,p_l,valid_l,e_l,g_n,v_n,b_l,og_l,_ = sess.run([self.local_AC.value_loss,
            self.local_AC.policy_loss,
            self.local_AC.valid_loss,
            self.local_AC.entropy,
            self.local_AC.grad_norms,
            self.local_AC.var_norms,
            self.local_AC.blocking_loss,
            self.local_AC.on_goal_loss,
            self.local_AC.apply_grads],
            feed_dict=feed_dict)
        return v_l/len(rollout), p_l/len(rollout), valid_l/len(rollout), e_l/len(rollout), b_l/len(rollout), og_l/len(rollout), g_n, v_n

    def shouldRun(self, coord, episode_count):
        if TRAINING:
            return (not coord.should_stop())
        else:
            return (episode_count < NUM_EXPS)

    
    def parse_path(self,path):
        '''needed function to take the path generated from M* and create the 
        observations and actions for the agent
        path: the exact path ouput by M*, assuming the correct number of agents
        returns: the list of rollouts for the "episode": 
                list of length num_agents with each sublist a list of tuples 
                (observation[0],observation[1],optimal_action,reward)'''
        result=[[] for i in range(num_workers)]
        goals = self.env.getGoals()
        # 
        agent_states=[]
        for i in range(num_workers):
            rnn_state = self.local_AC.state_init
            agent_states.append(rnn_state)
        # 


            
        for t in range(len(path[:-1])):
            observations=[]
            #
            inputs         = []
            goal_pos       = []
            visible_agents = []
            #
            move_queue=list(range(num_workers))
            for agent in range(1,num_workers+1):
                # observations.append(self.env._observe(agent))
                # 
                o = self.env._observe(agent)
                observations.append(o)
                inputs.append(o[0])
                goal_pos.append(o[1])
            #compute up to LSTM in parallel
            h3_vec = sess.run([self.local_AC.h3], 
                                            feed_dict={self.local_AC.inputs:inputs,
                                                        self.local_AC.goal_pos:goal_pos})
            h3_vec=h3_vec[0]
            rnn_out=[]
            #now go all the way past the lstm sequentially feeding the rnn_state
            for a in range(0,num_workers):
                rnn_state=agent_states[a]
                lstm_output,state_out, pred_priority = sess.run([self.local_AC.rnn_out,self.local_AC.state_out,self.local_AC.priority], 
                                            feed_dict={self.local_AC.inputs:[inputs[a]],
                                                        self.local_AC.h3:[h3_vec[a]],
                                                        self.local_AC.state_in[0]:rnn_state[0],
                                                        self.local_AC.state_in[1]:rnn_state[1]})
                rnn_out.append(lstm_output[0])
                agent_states[a]=state_out
                
                self.env.world.resetMessgeBuffer(a+1)
                self.env.world.setMessage(a+1, state_out[0][-1])
                visible_agents.append(self.env.getVisibleAgents(a+1, 11))
                # 


            steps=0
            # extract implicit priority each step for all agents
            while len(move_queue)>0:
                steps+=1
                i=move_queue.pop(0)
                o=observations[i]
                pos=path[t][i]
                newPos=path[t+1][i]#guaranteed to be in bounds by loop guard
                direction=(newPos[0]-pos[0],newPos[1]-pos[1])
                a=self.env.world.getAction(direction)
                state, reward, done, nextActions, on_goal, blocking, valid_action, priority=self.env._step((i+1,a),output_priority=True)
                next_poss = path[t+1]

                optimal_next_pos = []
                # bottom-right goal
                if goals[i][0]-pos[0] > 0 and goals[i][1] - pos[1] > 0:
                    optimal_next_pos = [(pos[0]+1, pos[1]), (pos[0], pos[1]+1)]
                # bottom-left goal
                elif goals[i][0] - pos[0] > 0 and goals[i][1] - pos[1] < 0:
                    optimal_next_pos = [(pos[0]+1, pos[1]), (pos[0], pos[1]-1)]
                # up-left goal
                elif goals[i][0] - pos[0] < 0 and goals[i][1] - pos[1] < 0:
                    optimal_next_pos = [(pos[0]-1, pos[1]), (pos[0], pos[1]-1)]
                # up-right goal
                elif goals[i][0] - pos[0] < 0 and goals[i][1] - pos[1] > 0:
                    optimal_next_pos = [(pos[0]-1, pos[1]), (pos[0], pos[1]+1)]
                # bottom goal
                elif goals[i][0] - pos[0] > 0:
                    optimal_next_pos = [(pos[0]+1, pos[1])]
                # up goal
                elif goals[i][0] - pos[0] < 0:
                    optimal_next_pos = [(pos[0]-1, pos[1])]
                # left goal
                elif goals[i][1] - pos[1] < 0:
                    optimal_next_pos = [(pos[0], pos[1]-1)]
                # right goal
                elif goals[i][1] - pos[1] > 0:
                    optimal_next_pos = [(pos[0], pos[1]+1)]

                priority = False
                # if stay in odrm*
                if a==0:
                    if len(optimal_next_pos) == 0:
                        priority = True
                elif a > 0:
                    if len(optimal_next_pos) == 1 and optimal_next_pos[0] == next_poss[i]:
                        priority = True
                    elif len(optimal_next_pos) ==2 and (optimal_next_pos[0] == next_poss[i] or optimal_next_pos[1] == next_poss[i]):
                        priority = True
                        
                
                self.env.world.setPriority(i+1, priority)
                
                if steps>num_workers**2:
                    #if we have a very confusing situation where lots of agents move
                    #in a circle (difficult to parse and also (mostly) impossible to learn)
                    return None
                if not valid_action:
                    #the tie must be broken here
                    move_queue.append(i)
                    continue

                result[i].append([o[0],o[1],a,priority,[0 for i in range(512)]])

            # construct decentralization communication according to extracted priority
            for aID in range(1,num_workers+1):
                level = self.env.world.getAgentLevel(aID)
                if level == 1:
                    self.env.checkHighLevel(aID,visible_agents[aID-1])

            for aID in range(1,num_workers+1):
                level = self.env.world.getAgentLevel(aID)
                if level == 0:
                    self.env.checkLowLevel(aID,visible_agents[aID-1])
                
            for aID in range(1,num_workers+1):
                level = self.env.world.getAgentLevel(aID)
                if level == -1:
                    self.env.checkUndefinedLevel(aID,visible_agents[aID-1])

            for aID in range(1,num_workers+1):
                level = self.env.world.getAgentLevel(aID)
                if level==1:
                    self.env.aggregateAndBoardcast(aID,visible_agents[aID-1])

            for aID in range(1,num_workers+1):
                self.env.reduceMessageBuffer(aID)
                message = self.env.world.getAggregateMessage(aID)
                result[aID-1][-1][-1] = message
            
        return result
        
    def work(self,max_episode_length,gamma,sess,coord,saver):
        global episode_count, swarm_reward, episode_rewards, episode_lengths, episode_mean_values, episode_invalid_ops,episode_wrong_blocking #, episode_invalid_goals
        global episode_total_move, episode_total_collision, episode_agent_collision, episode_success, episode_success_length, episode_success_total_move,episode_success_agent_collision, episode_success_total_collision
        global SEED
        total_steps, i_buf = 0, 0
        episode_buffers, s1Values = [ [] for _ in range(NUM_BUFFERS) ], [ [] for _ in range(NUM_BUFFERS) ]

        with sess.as_default(), sess.graph.as_default():
            while self.shouldRun(coord, episode_count):
                
                sess.run(self.pull_global)

                episode_buffer, episode_values = [], []
                episode_reward = episode_step_count = episode_inv_count = 0
                d = False

                # Initial state from the environment
                if self.agentID==1:
                    # SEED += 1
                    # seed_everything(SEED)
                    self.env._reset(self.agentID)
                print(f'{self.workerID} before synchronizing at 237 {episode_count}')
                self.synchronize() # synchronize starting time of the threads 
                # print(f'{self.workerID} finish synchronizing at 235 {episode_count}')
                
                validActions          = self.env._listNextValidActions(self.agentID)
                s                     = self.env._observe(self.agentID)
                blocking              = False
                p=self.env.world.getPos(self.agentID)
                on_goal               = self.env.world.goals[p[0],p[1]]==self.agentID
                # s                     = self.env._observe(self.agentID)
                rnn_state             = self.local_AC.state_init
                rnn_state0            = rnn_state
                priority         = self.env.world.getPriority(self.agentID)
                default_message          = np.array([0 for i in range(512)])
                RewardNb = 0 
                wrong_blocking  = 0
                wrong_on_goal=0

                if self.agentID==1:
                    global demon_probs
                    demon_probs[self.metaAgentID]=np.random.rand()
                print(f'{self.workerID} before synchronizing at 256 {episode_count}')
                self.synchronize() # synchronize starting time of the threadss 
                # print(f'{self.workerID} finish synchronizing at 254 {episode_count}')
                

                # reset swarm_reward (for tensorboard)
                swarm_reward[self.metaAgentID] = 0
                if episode_count<PRIMING_LENGTH or demon_probs[self.metaAgentID]<DEMONSTRATION_PROB:
                    #for the first PRIMING_LENGTH episodes, or with a certain probability
                    #don't train on the episode and instead observe a demonstration from M*
                    if self.workerID==1 and episode_count%100==0:
                        saver.save(sess, model_path+'/model-'+str(int(episode_count))+'.cptk')
                    global rollouts
                    rollouts[self.metaAgentID]=None
                    if(self.agentID==1):
                        world=self.env.getObstacleMap()
                        start_positions=tuple(self.env.getPositions())
                        goals=tuple(self.env.getGoals())
                        try:
                            # odrm*
                            mstar_path=cpp_mstar.find_path(world,start_positions,goals,2,5)

                            rollouts[self.metaAgentID]=self.parse_path(mstar_path)
                            print("solved",episode_count)
                        except OutOfTimeError:
                            #M* timed out 
                            print("timeout",episode_count)
                        except NoSolutionError:
                            print("nosol????",episode_count,start_positions)
                    print(f'{self.workerID} before synchronizing at 287 {episode_count}')
                    self.synchronize() # synchronize threads
                    # print(f'{self.workerID} finish synchronizing at 280 {episode_count}')
                    if rollouts[self.metaAgentID] is not None:
                        bc_l,pi_l,i_l=self.train(rollouts[self.metaAgentID][self.agentID-1], sess, gamma, None, rnn_state0, imitation=True)
                        episode_count+=1./num_workers
                        if self.agentID==1:
                            summary = tf.Summary()
                            summary.value.add(tag='Losses/Behavior Cloning loss', simple_value=bc_l)
                            summary.value.add(tag='Losses/Priority loss', simple_value=pi_l)
                            summary.value.add(tag='Losses/Imitation loss', simple_value=i_l)
                            global_summary.add_summary(summary, int(episode_count))
                            global_summary.flush()
                        continue
                    continue
                saveGIF = False
                if OUTPUT_GIFS and self.workerID == 1 and ((not TRAINING) or (episode_count >= self.nextGIF)):
                    saveGIF = True
                    self.nextGIF =episode_count + 64
                    GIF_episode = int(episode_count)
                    episode_frames = [ self.env._render(mode='rgb_array',screen_height=900,screen_width=900) ]
                    
                while (not self.env.finished):
                    #Take an action using probabilities from policy network output.
                    a_dist,v,rnn_state,pred_blocking,pred_on_goal,pred_priority = sess.run([self.local_AC.policy,
                                                   self.local_AC.value,
                                                   self.local_AC.state_out,
                                                   self.local_AC.blocking,
                                                    self.local_AC.on_goal,
                                                    self.local_AC.priority], 
                                         feed_dict={self.local_AC.inputs:[s[0]],
                                                    self.local_AC.goal_pos:[s[1]],
                                                    self.local_AC.message:[default_message],
                                                    self.local_AC.state_in[0]:rnn_state[0],
                                                    self.local_AC.state_in[1]:rnn_state[1]})
                    # communication
                    self.env.world.resetMessgeBuffer(self.agentID)
                    self.env.world.setMessage(self.agentID, rnn_state[0][-1])
                    # priority = pred_priority.flatten()[0] > 0.5
                    self.env.world.setPriority(self.agentID, pred_priority)
                    visible_agents = self.env.getVisibleAgents(self.agentID, 10)
                    self.synchronize() # synchronize threads  
                    level = self.env.world.getAgentLevel(self.agentID)
                    if level == 1:
                        self.env.checkHighLevel(self.agentID,visible_agents)
                    self.synchronize() # synchronize threads  
                    level = self.env.world.getAgentLevel(self.agentID)
                    if level == 0:
                        self.env.checkLowLevel(self.agentID,visible_agents)
                    self.synchronize() # synchronize threads  
                    level = self.env.world.getAgentLevel(self.agentID)
                    if level == -1:
                        self.env.checkUndefinedLevel(self.agentID,visible_agents)
                    self.synchronize() # synchronize threads  
                    level = self.env.world.getAgentLevel(self.agentID)
                    if level==1:
                        self.env.aggregateAndBoardcast(self.agentID,visible_agents)
                    
                    self.synchronize() # synchronize threads 
                    self.env.reduceMessageBuffer(self.agentID)
                    message = self.env.world.getAggregateMessage(self.agentID)
                    # message = default_message
                    self.synchronize() # synchronize threads  


                    a_dist,v,rnn_state,pred_blocking,pred_on_goal,pred_priority = sess.run([self.local_AC.policy,
                                                   self.local_AC.value,
                                                   self.local_AC.state_out,
                                                   self.local_AC.blocking,
                                                    self.local_AC.on_goal,
                                                    self.local_AC.priority], 
                                         feed_dict={self.local_AC.inputs:[s[0]],
                                                    self.local_AC.goal_pos:[s[1]],
                                                    self.local_AC.message:[message],
                                                    self.local_AC.state_in[0]:rnn_state[0],
                                                    self.local_AC.state_in[1]:rnn_state[1]})
                    if(not (np.argmax(a_dist.flatten()) in validActions)):
                        episode_inv_count += 1
                    train_valid = np.zeros(a_size)
                    train_valid[validActions] = 1

                    valid_dist = np.array([a_dist[0,validActions]])
                    valid_dist /= np.sum(valid_dist)
                    # priority = pred_priority.flatten()[0] > 0.5

                    if TRAINING:
                        if (pred_blocking.flatten()[0] < 0.5) == blocking:
                            wrong_blocking += 1
                        if (pred_on_goal.flatten()[0] < 0.5) == on_goal:
                            wrong_on_goal += 1
                        a           = validActions[ np.random.choice(range(valid_dist.shape[1]),p=valid_dist.ravel()) ]
                        # if not priority:
                        #     a = 0
                        train_val   = 1.
                    else:
                        a         = np.argmax(a_dist.flatten())
                        if a not in validActions or not GREEDY:
                            a     = validActions[ np.random.choice(range(valid_dist.shape[1]),p=valid_dist.ravel()) ]
                        # if not priority:
                        #     a = 0
                        train_val = 1.
                    
                    _, r, _, _, on_goal,blocking,_,priority = self.env._step((self.agentID, a),episode=episode_count,output_priority=True)
                    print(f'{self.workerID} before synchronizing at 341 {episode_count}')
                    self.synchronize() # synchronize threads
                    # print(f'{self.workerID} finish synchronizing at 334 {episode_count}')

                    # Get common observation for all agents after all individual actions have been performed
                    s1           = self.env._observe(self.agentID)
                    validActions = self.env._listNextValidActions(self.agentID, a,episode=episode_count)
                    d            = self.env.finished

                    if saveGIF:
                        episode_frames.append(self.env._render(mode='rgb_array',screen_width=900,screen_height=900))
                    # replay buffer content
                    episode_buffer.append([s[0],a,r,s1,d,v[0,0],train_valid,pred_on_goal,int(on_goal),pred_blocking,int(blocking),message,s[1],train_val])
                    episode_values.append(v[0,0])
                    episode_reward += r
                    s = s1
                    total_steps += 1
                    episode_step_count += 1

                    if r>0:
                        RewardNb += 1
                    if d == True:
                        print('\n{} Goodbye World. We did it!'.format(episode_step_count), end='\n')
                    else:
                        print('\n{} Hello World. We are trying!'.format(episode_step_count), end='\n')

                    # If the episode hasn't ended, but the experience buffer is full, then we
                    # make an update step using that experience rollout.
                    if TRAINING and (len(episode_buffer) % EXPERIENCE_BUFFER_SIZE == 0 or d):
                        # Since we don't know what the true final return is, we "bootstrap" from our current value estimation.
                        if len(episode_buffer) >= EXPERIENCE_BUFFER_SIZE:
                            episode_buffers[i_buf] = episode_buffer[-EXPERIENCE_BUFFER_SIZE:]
                        else:
                            episode_buffers[i_buf] = episode_buffer[:]

                        if d:
                            s1Values[i_buf] = 0
                        else:
                            s1Values[i_buf] = sess.run(self.local_AC.value, 
                                 feed_dict={self.local_AC.inputs:np.array([s[0]])
                                            ,self.local_AC.goal_pos:[s[1]]
                                            ,self.local_AC.state_in[0]:rnn_state[0]
                                            ,self.local_AC.state_in[1]:rnn_state[1]})[0,0]

                        if (episode_count-EPISODE_START) < NUM_BUFFERS:
                            i_rand = np.random.randint(i_buf+1)
                        else:
                            i_rand = np.random.randint(NUM_BUFFERS)
                            tmp = np.array(episode_buffers[i_rand])
                            while tmp.shape[0] == 0:
                                i_rand = np.random.randint(NUM_BUFFERS)
                                tmp = np.array(episode_buffers[i_rand])
                        v_l,p_l,valid_l,e_l,b_l,og_l,g_n,v_n = self.train(episode_buffers[i_rand],sess,gamma,s1Values[i_rand],rnn_state0)

                        i_buf = (i_buf + 1) % NUM_BUFFERS
                        rnn_state0             = rnn_state
                        episode_buffers[i_buf] = []

                    print(f'{self.workerID} before synchronizing at 397 {episode_count}')
                    self.synchronize() # synchronize threads
                    # print(f'{self.workerID} finish synchronizing at 390 {episode_count}')
                    # sess.run(self.pull_global)
                    if episode_step_count >= max_episode_length or d:
                        break

                total_move, collision_total, collision_agent, collision_static = self.env.getMetrics()

                cur_success = 0
                if d:
                    cur_success = 1
                    episode_success_length[self.metaAgentID].append(episode_step_count)
                    episode_success_total_move[self.metaAgentID].append(total_move)
                    episode_success_agent_collision[self.metaAgentID].append(collision_agent)
                    episode_success_total_collision[self.metaAgentID].append(collision_total)
                episode_lengths[self.metaAgentID].append(episode_step_count)
                episode_mean_values[self.metaAgentID].append(np.nanmean(episode_values))
                episode_invalid_ops[self.metaAgentID].append(episode_inv_count)
                episode_wrong_blocking[self.metaAgentID].append(wrong_blocking)

                episode_total_move[self.metaAgentID].append(total_move)
                episode_total_collision[self.metaAgentID].append(collision_total)
                episode_agent_collision[self.metaAgentID].append(collision_agent)
                episode_success[self.metaAgentID].append(cur_success)


                # Periodically save gifs of episodes, model parameters, and summary statistics.
                if episode_count % EXPERIENCE_BUFFER_SIZE == 0 and printQ:
                    print('                                                                                   ', end='\r')
                    print('{} Episode terminated ({},{})'.format(episode_count, self.agentID, RewardNb), end='\r')

                swarm_reward[self.metaAgentID] += episode_reward
                print(f'{self.workerID} before synchronizing at 415 {episode_count}')
                self.synchronize() # synchronize threads
                # print(f'{self.workerID} finish synchronizing at 408 {episode_count}')
                episode_rewards[self.metaAgentID].append(swarm_reward[self.metaAgentID])

                
                print(f'worker {self.workerID} finished episode {episode_count} INFO-{INFO}')

                if not TRAINING:
                    mutex.acquire()
                    if episode_count < NUM_EXPS:
                        plan_durations[episode_count] = episode_step_count
                    if self.workerID == 1:
                        episode_count += 1
                        print('({}) Thread {}: {} steps, {:.2f} reward ({} invalids).'.format(episode_count, self.workerID, episode_step_count, episode_reward, episode_inv_count))
                    GIF_episode = int(episode_count)
                    mutex.release()
                else:
                    # print(f'{self.workerID} before episode_count({episode_count}) added')
                    # mutex.acquire()
                    episode_count+=1./num_workers

                    if episode_count % SUMMARY_WINDOW == 0:
                        if episode_count % 100 == 0:
                            print ('Saving Model', end='\n')
                            saver.save(sess, model_path+'/model-'+str(int(episode_count))+'.cptk')
                            print ('Saved Model', end='\n')
                        SL = SUMMARY_WINDOW * num_workers
                        mean_reward = np.nanmean(episode_rewards[self.metaAgentID][-SL:])
                        mean_length = np.nanmean(episode_lengths[self.metaAgentID][-SL:])
                        mean_value = np.nanmean(episode_mean_values[self.metaAgentID][-SL:])
                        mean_invalid = np.nanmean(episode_invalid_ops[self.metaAgentID][-SL:])
                        mean_wrong_blocking = np.nanmean(episode_wrong_blocking[self.metaAgentID][-SL:])
                        mean_total_move = np.nanmean(episode_total_move[self.metaAgentID][-SL:])
                        mean_total_collision = np.nanmean(episode_total_collision[self.metaAgentID][-SL:])
                        mean_agent_collision = np.nanmean(episode_agent_collision[self.metaAgentID][-SL:])
                        mean_success = np.nanmean(episode_success[self.metaAgentID][-SL:])
                        mean_success_length = np.nanmean(episode_success_length[self.metaAgentID][-SL:])
                        mean_success_total_move = np.nanmean(episode_success_total_move[self.metaAgentID][-SL:])
                        mean_success_agent_collision = np.nanmean(episode_success_agent_collision[self.metaAgentID][-SL:])
                        mean_success_total_collision = np.nanmean(episode_success_total_collision[self.metaAgentID][-SL:])
                        current_learning_rate = sess.run(lr,feed_dict={global_step:episode_count})

                        summary = tf.Summary()
                        summary.value.add(tag='Perf/Learning Rate',simple_value=current_learning_rate)
                        summary.value.add(tag='Perf/Reward', simple_value=mean_reward)
                        summary.value.add(tag='Perf/Length', simple_value=mean_length)
                        summary.value.add(tag='Perf/Valid Rate', simple_value=(mean_length-mean_invalid)/mean_length)
                        summary.value.add(tag='Perf/Blocking Prediction Accuracy', simple_value=(mean_length-mean_wrong_blocking)/mean_length)
                        summary.value.add(tag='Perf/Total Move', simple_value=mean_total_move)
                        summary.value.add(tag='Perf/Total Collision', simple_value=mean_total_collision)
                        summary.value.add(tag='Perf/Agent Collision', simple_value=mean_agent_collision)
                        summary.value.add(tag='Perf/Success', simple_value=mean_success)
                        summary.value.add(tag='Perf/Success Length', simple_value=mean_success_length)
                        summary.value.add(tag='Perf/Success Total Move', simple_value=mean_success_total_move)
                        summary.value.add(tag='Perf/Success Agent Collision', simple_value=mean_success_agent_collision)
                        summary.value.add(tag='Perf/Success Total Collision', simple_value=mean_success_total_collision)


                        summary.value.add(tag='Losses/Value Loss', simple_value=v_l)
                        summary.value.add(tag='Losses/Policy Loss', simple_value=p_l)
                        summary.value.add(tag='Losses/Blocking Loss', simple_value=b_l)
                        summary.value.add(tag='Losses/On Goal Loss', simple_value=og_l)
                        summary.value.add(tag='Losses/Valid Loss', simple_value=valid_l)
                        summary.value.add(tag='Losses/Grad Norm', simple_value=g_n)
                        summary.value.add(tag='Losses/Var Norm', simple_value=v_n)
                        global_summary.add_summary(summary, int(episode_count))

                        global_summary.flush()

                        if printQ:
                            print('{} Tensorboard updated ({})'.format(episode_count, self.workerID), end='\r')
                    # mutex.release()
                    print('({}) Thread {}: {} steps).'.format(episode_count, self.workerID, episode_step_count))

                if saveGIF:
                    # Dump episode frames for external gif generation (otherwise, makes the jupyter kernel crash)
                    time_per_step = 0.1
                    images = np.array(episode_frames)
                    if TRAINING:
                        make_gif(images, '{}/episode_{:d}_{:d}_{:.1f}.gif'.format(gifs_path,GIF_episode,episode_step_count,swarm_reward[self.metaAgentID]))
                    else:
                        make_gif(images, '{}/episode_{:d}_{:d}.gif'.format(gifs_path,GIF_episode,episode_step_count), duration=len(images)*time_per_step,true_image=True,salience=False)
                if SAVE_EPISODE_BUFFER:
                    with open('gifs3D/episode_{}.dat'.format(GIF_episode), 'wb') as file:
                        pickle.dump(episode_buffer, file)

# ## Training

# In[ ]:
from tfdeterminism import patch
patch()


os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['HOROVOD_FUSION_THRESHOLD']='0'
SEED = 1804
seed_everything(SEED)
# Learning parameters
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
DEMONSTRATION_PROB     = 0.5 # probability of training on a demonstration per episode

REWARD_MODIFIED = False

INFO=f'initTrain_Obs11_G1020_OD03_A8/ACNetCommold_full_Clip1000_LR5'
# INFO = 'comm-debug'


    

model_path             = f'model/'+INFO
load_model_path        = f'model/ACNetCommold_full_Clip1000_LR5_best'
train_path             = f'train_log/'+INFO+'/'+TIMESTAMP
# Simulation options
FULL_HELP              = False
OUTPUT_GIFS            = False
SAVE_EPISODE_BUFFER    = False

# Testing
TRAINING               = True
GREEDY                 = False
NUM_EXPS               = 100
MODEL_NUMBER           = 313000

# Shared arrays for tensorboard
episode_rewards        = [ [] for _ in range(NUM_META_AGENTS) ]
episode_lengths        = [ [] for _ in range(NUM_META_AGENTS) ]
episode_mean_values    = [ [] for _ in range(NUM_META_AGENTS) ]
episode_invalid_ops    = [ [] for _ in range(NUM_META_AGENTS) ]
episode_wrong_blocking = [ [] for _ in range(NUM_META_AGENTS) ]
episode_total_move           = [ [] for _ in range(NUM_META_AGENTS) ]
episode_total_collision      = [ [] for _ in range(NUM_META_AGENTS) ]
episode_agent_collision      = [ [] for _ in range(NUM_META_AGENTS) ]
episode_success              = [ [] for _ in range(NUM_META_AGENTS) ]
episode_success_length          = [ [] for _ in range(NUM_META_AGENTS) ]
episode_success_total_move      = [ [] for _ in range(NUM_META_AGENTS) ]
episode_success_agent_collision = [ [] for _ in range(NUM_META_AGENTS) ]
episode_success_total_collision = [ [] for _ in range(NUM_META_AGENTS) ]
rollouts               = [ None for _ in range(NUM_META_AGENTS)]
demon_probs=[np.random.rand() for _ in range(NUM_META_AGENTS)]
printQ                 = False # (for headless)
swarm_reward           = [0]*NUM_META_AGENTS


# In[ ]:


tf.reset_default_graph()
print("Hello World")
if not os.path.exists(model_path):
    os.makedirs(model_path)
config = tf.ConfigProto(allow_soft_placement = True)
config.gpu_options.allow_growth=True

# mutex = threading.Lock()
if not TRAINING:
    plan_durations = np.array([0 for _ in range(NUM_EXPS)])
    mutex = threading.Lock()
    gifs_path += '_tests'
    if SAVE_EPISODE_BUFFER and not os.path.exists('gifs3D'):
        os.makedirs('gifs3D')

#Create a directory to save episode playback gifs to
if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)

with tf.device("/gpu:0"):
    master_network = ACNet(GLOBAL_NET_SCOPE,a_size,None,False,GRID_SIZE,GLOBAL_NET_SCOPE) # Generate global network
    # global_vars        = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, GLOBAL_NET_SCOPE+'/qvalues')
    # finetune_vars      = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, GLOBAL_NET_SCOPE+'/qvalues/finetune')
    # pdb.set_trace()

    global_step = tf.placeholder(tf.float32)
    if ADAPT_LR:
        #computes LR_Q/sqrt(ADAPT_COEFF*steps+1)
        #we need the +1 so that lr at step 0 is defined
        lr=tf.divide(tf.constant(LR_Q),tf.sqrt(tf.add(1.,tf.multiply(tf.constant(ADAPT_COEFF),global_step))))
        # lr_ft=tf.divide(tf.constant(LR_Q_finetune),tf.sqrt(tf.add(1.,tf.multiply(tf.constant(ADAPT_COEFF),global_step))))
    else:
        lr=tf.constant(LR_Q)
        # lr_ft=tf.constant(LR_Q_finetune)
    trainer = tf.contrib.opt.NadamOptimizer(learning_rate=lr, use_locking=True)
    # trainer_ft= tf.contrib.opt.NadamOptimizer(learning_rate=lr_ft,use_locking=True)

    if TRAINING:
        num_workers = NUM_THREADS # Set workers # = # of available CPU threads
    else:
        num_workers = NUM_THREADS
        NUM_META_AGENTS = 1
    
    gameEnvs, workers, groupLocks = [], [], []
    n=1#counter of total number of agents (for naming)
    for ma in range(NUM_META_AGENTS):
        num_agents=NUM_THREADS
        gameEnv = mapf_gym.MAPFEnv(num_agents=num_agents, DIAGONAL_MOVEMENT=DIAG_MVMT, SIZE=ENVIRONMENT_SIZE, 
                                        observation_size=GRID_SIZE,PROB=OBSTACLE_DENSITY, FULL_HELP=FULL_HELP)
        gameEnvs.append(gameEnv)

        # Create groupLock
        workerNames = ["worker_"+str(i) for i in range(n,n+num_workers)]
        groupLock = GroupLock.GroupLock([workerNames,workerNames])
        groupLocks.append(groupLock)

        # Create worker classes
        workersTmp = []
        for i in range(ma*num_workers+1,(ma+1)*num_workers+1):
            workersTmp.append(Worker(gameEnv,ma,n,a_size,groupLock,MAX_EPISODE))
            n+=1
        workers.append(workersTmp)

    global_summary = tf.summary.FileWriter(train_path)
    saver = tf.train.Saver(max_to_keep=2)

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        if load_model == True:
            print ('Loading Model...')
            if not TRAINING:
                with open(load_model_path+'/checkpoint', 'w') as file:
                    file.write('model_checkpoint_path: "model-{}.cptk"'.format(MODEL_NUMBER))
                    file.close()
            ckpt = tf.train.get_checkpoint_state(load_model_path)
            episode_count = 0
            saver.restore(sess,ckpt.model_checkpoint_path)
            print("episode_count set to ",episode_count)
            if RESET_TRAINER:
                trainer = tf.contrib.opt.NadamOptimizer(learning_rate=lr, use_locking=True)

        # This is where the asynchronous magic happens.
        # Start the "work" process for each worker in a separate thread.
        worker_threads = []
        for ma in range(NUM_META_AGENTS):
            for worker in workers[ma]:
                groupLocks[ma].acquire(0,worker.name) # synchronize starting time of the threads
                worker_work = lambda: worker.work(max_episode_length,gamma,sess,coord,saver)
                print("Starting worker " + str(worker.workerID))
                t = threading.Thread(target=(worker_work))
                t.start()
                worker_threads.append(t)
        print('main get a ______________________________________')
        coord.join(worker_threads)
        print('main get b ______________________________________')

if not TRAINING:
    print([np.mean(plan_durations), np.sqrt(np.var(plan_durations)), np.mean(np.asarray(plan_durations < max_episode_length, dtype=float))])


# In[ ]:




