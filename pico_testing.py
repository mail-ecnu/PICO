import tensorflow as tf
from ACNetComm_old import ACNet
import numpy as np
import json
import os
import mapf_gym as mapf_gym
import time
from od_mstar3 import cpp_mstar
from od_mstar3.col_set_addition import OutOfTimeError,NoSolutionError

import pdb

prepath = 'test_result/'
model_path = 'ACNetCommold_full_Clip1000_LR5_best'
result_name = 'pico'
results_path=f"{prepath}{result_name}_result_random"
if model_path == '':
    results_path=f"{prepath}odrm_result_random"
environment_path="saved_environments"
if not os.path.exists(results_path):
    os.makedirs(results_path)

class PICO(object):
    '''
    This class provides functionality for running multiple instances of the 
    trained network in a single environment
    '''
    def __init__(self,model_path1,grid_size):
        self.grid_size=grid_size
        config = tf.ConfigProto(allow_soft_placement = True)
        config.gpu_options.allow_growth=True
        self.sess=tf.Session(config=config)
        self.network=ACNet("global",5,None,False,grid_size,"global")
        
        # var = tf.global_variables()
        #load the weights from the checkpoint (only the global ones!)
        ckpt = tf.train.get_checkpoint_state(model_path1)
        # var_dict = tf.train.list_variables(ckpt.model_checkpoint_path)
        # var_dict = tf.trainable_variables()
        # var_to_restore = []
        # for val in var_dict:
        #     if 'global' in val.name:
        #         if 'finetune' not in val.name:
        #             var_to_restore.append(val)
        #         print(val.name)
        # pdb.set_trace()
        saver = tf.train.Saver()
        saver.restore(self.sess,ckpt.model_checkpoint_path)
        
        # pdb.set_trace()
        
    def set_env(self,gym):
        self.num_agents=gym.num_agents
        self.agent_states=[]
        for i in range(self.num_agents):
            rnn_state = self.network.state_init
            self.agent_states.append(rnn_state)
        self.size=gym.SIZE
        self.env=gym
        
    def step_all_parallel(self):
        action_probs=[None for i in range(self.num_agents)]
        '''advances the state of the environment by a single step across all agents'''
        #parallel inference
        actions=[]
        inputs=[]
        goal_pos=[]
        for agent in range(1,self.num_agents+1):
            o=self.env._observe(agent)
            inputs.append(o[0])
            goal_pos.append(o[1])
        #compute up to LSTM in parallel
        h3_vec = self.sess.run([self.network.h3], 
                                         feed_dict={self.network.inputs:inputs,
                                                    self.network.goal_pos:goal_pos})
        h3_vec=h3_vec[0]
        rnn_out=[]
        #now go all the way past the lstm sequentially feeding the rnn_state
        for a in range(0,self.num_agents):
            rnn_state=self.agent_states[a]
            lstm_output,state = self.sess.run([self.network.rnn_out,self.network.state_out], 
                                         feed_dict={self.network.inputs:[inputs[a]],
                                                    self.network.h3:[h3_vec[a]],
                                                    self.network.state_in[0]:rnn_state[0],
                                                    self.network.state_in[1]:rnn_state[1]})
            rnn_out.append(lstm_output[0])
            self.agent_states[a]=state

            own_message = state[0][-1]
            self.env.world.resetMessgeBuffer(a+1)
            self.env.world.setMessage(a+1, own_message)

        #now finish in parallel
        priority_vec=self.sess.run([self.network.priority], 
                                         feed_dict={self.network.rnn_out:rnn_out})
        pred_priority = priority_vec[0]
        visible_agents_list = []
        for a in range(0,self.num_agents):    
            # priority = pred_priority[a].flatten()[0] > 0.5
            self.env.world.setPriority(a+1, pred_priority[a])
            visible_agents = self.env.getVisibleAgents(a+1, 10)
            visible_agents_list.append(visible_agents)
        for a in range(0,self.num_agents):  
            level = self.env.world.getAgentLevel(a+1)
            if level == 1:
                self.env.checkHighLevel(a+1,visible_agents_list[a])
        for a in range(0,self.num_agents):    
            level = self.env.world.getAgentLevel(a+1)
            if level == 0:
                self.env.checkLowLevel(a+1,visible_agents_list[a])
        for a in range(0,self.num_agents):   
            level = self.env.world.getAgentLevel(a+1)
            if level == -1:
                self.env.checkUndefinedLevel(a+1,visible_agents_list[a])
        for a in range(0,self.num_agents):  
            level = self.env.world.getAgentLevel(a+1)
            if level==1:
                self.env.aggregateAndBoardcast(a+1,visible_agents_list[a])
        messages = []
        for a in range(0,self.num_agents):
            self.env.reduceMessageBuffer(a+1)
            message = self.env.world.getAggregateMessage(a+1)
            messages.append(message)
            # agent_level.append(self.env.getCommMetrics())
        policy_vec=self.sess.run([self.network.policy], 
                                         feed_dict={self.network.rnn_out:rnn_out,
                                                    self.network.message:messages})
        policy_vec=policy_vec[0]
        for agent in range(1,self.num_agents+1):
            action=np.argmax(policy_vec[agent-1])
            self.env._step((agent,action))

        return np.array(messages)
        
    def find_path(self,max_step=256):
        '''run a full environment to completion, or until max_step steps'''
        solution=[]
        step=0
        messages_solution = []
        agent_level_solution = []
        while((not self.env._complete()) and step<max_step):
            timestep=[]
            for agent in range(1,self.env.num_agents+1):
                timestep.append(self.env.world.getPos(agent))
            solution.append(np.array(timestep))
            message = self.step_all_parallel()
            agent_level = self.env.getCommMetrics()
            agent_level_solution.append(agent_level)
            messages_solution.append(message)
            # agent_level_solution.append(agent_level)
            step+=1
            #print(step)
        if step==max_step:
            raise OutOfTimeError
        for agent in range(1,self.env.num_agents):
            timestep.append(self.env.world.getPos(agent))
        return np.array(solution),np.array(messages_solution),np.array(agent_level_solution)
    
def make_name(n,s,d,id,extension,dirname,extra=""):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    if extra=="":
        return dirname+'/'+"{}_agents_{}_size_{}_density_id_{}{}".format(n,s,d,id,extension)
    else:
        return dirname+'/'+"{}_agents_{}_size_{}_density_id_{}_{}{}".format(n,s,d,id,extra,extension)
    
def run_simulations(next,pico):
    #txt file: planning time, crash, nsteps, finished
    (n,s,d,id) = next
    environment_data_filename=make_name(n,s,d,id,".npy",environment_path,extra="environment")
    world=np.load(environment_data_filename)
    gym=mapf_gym.MAPFEnv(num_agents=n, world0=world[0],goals0=world[1])
    if pico:
        pico.set_env(gym)
    solution_filename=make_name(n,s,d,id,".npy",results_path+f'/w{s}_od{d}_a{num_agents}',extra="solution")
    message_filename=make_name(n,s,d,id,".npy",results_path+f'/w{s}_od{d}_a{num_agents}',extra="message")
    agent_level_filename=make_name(n,s,d,id,".npy",results_path+f'/w{s}_od{d}_a{num_agents}',extra="agent_level")
    txt_filename=make_name(n,s,d,id,".txt",results_path+f'/w{s}_od{d}_a{num_agents}')
    world=gym.getObstacleMap()
    start_positions=tuple(gym.getPositions())
    goals=tuple(gym.getGoals())
    start_time=time.time()
    results=dict()
    start_time=time.time()
    try:
        print(f'{result_name}')
        print('Starting test ({},{},{},{})'.format(n,s,d,id))
        if pico:
            path, messages, agent_level=pico.find_path(256 + 128*int(s>=80) + 128*int(s>=160))
        else:
            world=gym.getObstacleMap()
            start_positions=tuple(gym.getPositions())
            goals=tuple(gym.getGoals())
            path=cpp_mstar.find_path(world,start_positions,goals,2,5)
        results['finished']=True
        results['time']=time.time()-start_time
        results['length']=len(path)
        results['total_move'] = gym.total_move
        results['collision_total'] = gym.collision_total
        results['collision_static'] = gym.collision_static
        results['collision_agent'] = gym.collision_agent
        np.save(solution_filename,path)
        np.save(message_filename,messages)
        np.save(agent_level_filename,agent_level)
        print(f'success-len:{len(path)}-tm:{gym.total_move}-coli:{gym.collision_total}-{gym.collision_agent}')
    except OutOfTimeError:
        results['time']=time.time()-start_time
        results['finished']=False
        results['length']=256
        results['total_move'] = gym.total_move
        results['collision_total'] = gym.collision_total
        results['collision_static'] = gym.collision_static
        results['collision_agent'] = gym.collision_agent
        print(f'tm:{gym.total_move}-coli:{gym.collision_total}-{gym.collision_agent}-failed')
    results['crashed']=False
    f=open(txt_filename,'w')
    f.write(json.dumps(results))
    f.close()

    return results['finished'], results

if __name__ == "__main__":
#    import sys
#    num_agents = int(sys.argv[1])
    if model_path == '':
        pico=None
    else:
        pico=PICO('model/'+model_path,11)
    # num_agents_list = [8]
    num_agents_list = [8,16,32,64]
    size = 20
    density_list = [0,0.1,0.2,0.3]
    for num_agents in num_agents_list:
        if size==10 and num_agents>32:continue
        if size==20 and num_agents>128:continue
        if size==40 and num_agents>512:continue
        for density in density_list:
            # success_count = 0
            results = dict()
            suces   = dict()
            results['finished']=0
            results['length']=[]
            results['total_move'] = []
            results['collision_total'] = []
            results['collision_static'] = []
            results['collision_agent'] = []
            results['collision_rate'] = []

            suces['finished']=0
            suces['length']=[]
            suces['total_move'] = []
            suces['collision_total'] = []
            suces['collision_static'] = []
            suces['collision_agent'] = []
            suces['collision_rate'] = []
            for iter in range(100):
                print(f'running {num_agents_list}')
                # txt_filename=make_name(num_agents,size,density,iter,".txt",results_path+f'/w{size}_od{density}_a{num_agents}')
                # f=open(txt_filename,'r')
                # data = json.load(f)
                # f.close()
                # if data['finished'] == False:
                #     continue
                success, res = run_simulations((num_agents,size,density,iter),pico)
                if success:
                    # success_count += 1
                    suces['finished']+=success
                    suces['length'].append(res['length'])
                    suces['total_move'].append(res['total_move'])
                    suces['collision_total'].append(res['collision_total'])
                    suces['collision_static'].append(res['collision_static'])
                    suces['collision_agent'].append(res['collision_agent'])
                    suces['collision_rate'].append(res['collision_agent']/res['length'])
                results['finished']+=success
                results['length'].append(res['length'])
                results['total_move'].append(res['total_move'])
                results['collision_total'].append(res['collision_total'])
                results['collision_static'].append(res['collision_static'])
                results['collision_agent'].append(res['collision_agent'])
                results['collision_rate'].append(res['collision_agent']/res['length'])
                # print(f'success:{success_count}/100')
        
            f=open(results_path+f'/w{size}_od{density}_a{num_agents}_success.txt','w')
            f.write(json.dumps(suces))
            f.close()

            f=open(results_path+f'/w{size}_od{density}_a{num_agents}_all.txt','w')
            f.write(json.dumps(results))
            f.close()
                
print("finished all tests!")
