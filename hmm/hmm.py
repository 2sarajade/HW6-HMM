import numpy as np
class HiddenMarkovModel:
    """
    Class for Hidden Markov Model 
    """

    def __init__(self, observation_states: np.ndarray, hidden_states: np.ndarray, prior_p: np.ndarray, transition_p: np.ndarray, emission_p: np.ndarray):
        """

        Initialization of HMM object

        Args:
            observation_states (np.ndarray): observed states 
            hidden_states (np.ndarray): hidden states 
            prior_p (np.ndarray): prior probabities of hidden states 
            transition_p (np.ndarray): transition probabilites between hidden states
            emission_p (np.ndarray): emission probabilites from transition to hidden states 
        """             
        
        self.observation_states = observation_states
        self.observation_states_dict = {state: index for index, state in enumerate(list(self.observation_states))}

        self.hidden_states = hidden_states
        self.hidden_states_dict = {index: state for index, state in enumerate(list(self.hidden_states))}
        
        self.prior_p= prior_p
        self.transition_p = transition_p
        self.emission_p = emission_p

        if self.prior_p.shape != self.hidden_states.shape:
            raise ValueError("priors and states do not match")
        if abs(sum(self.prior_p) - 1) > .00000000001:
            raise ValueError("priors are not scaled properly")
        

    def forward(self, input_observation_states: np.ndarray) -> float:
        """
        TODO 

        This function runs the forward algorithm on an input sequence of observation states

        Args:
            input_observation_states (np.ndarray): observation sequence to run forward algorithm on 

        Returns:
            forward_probability (float): forward probability (likelihood) for the input observed sequence  
        """        
        
        # Step 1. Initialize variables
        num_in = len(input_observation_states)
        num_states = len(self.hidden_states)
        alpha = np.zeros((num_in, num_states))
        
       
        # Step 2. Calculate probabilities
        obs = self.observation_states_dict.get(input_observation_states[0])
        alpha[0] = [self.prior_p[i] * self.emission_p[i, obs] for i in range(num_states)]

        for i in range(1, num_in):
            for j in range(num_states):
                obs = self.observation_states_dict.get(input_observation_states[i])
                alpha[i,j] = np.sum([alpha[i-1, prev_state] * self.transition_p [prev_state, j] * self.emission_p[j,obs] for prev_state in range(num_states)])

        # Step 3. Return final probability 
        return np.sum(alpha[-1])
        


    def viterbi(self, decode_observation_states: np.ndarray) -> list:
        """
        TODO

        This function runs the viterbi algorithm on an input sequence of observation states

        Args:
            decode_observation_states (np.ndarray): observation state sequence to decode 

        Returns:
            best_hidden_state_sequence(list): most likely list of hidden states that generated the sequence observed states

        much of the code here is based on the tutorial linked in the HW6 instruction docs here:
        https://pieriantraining.com/viterbi-algorithm-implementation-in-python-a-practical-guide/
        """        
        
        # Step 1. Initialize variables
        #store probabilities of hidden state at each step 
        viterbi_table = np.zeros((len(decode_observation_states), len(self.hidden_states)))
        #store best path for traceback
        # best_path = np.zeros(len(decode_observation_states))  
        backpointer_table = np.zeros((len(decode_observation_states), len(self.hidden_states)), dtype=np.int8)       
        
        num_obs = len(decode_observation_states)
        num_states = len(self.hidden_states)

       # Step 2. Calculate Probabilities
        obs = self.observation_states_dict.get(decode_observation_states[0])
        viterbi_table[0] = [self.prior_p[i] * self.emission_p[i, obs] for i in range(num_states)]

        for i in range(1, num_obs):
            for j in range (num_states):
                obs = self.observation_states_dict.get(decode_observation_states[i])
                max_prob = np.max([viterbi_table[i-1, prev_state] * self.transition_p [prev_state, j] for prev_state in range(num_states)])
                viterbi_table[i,j] = max_prob *  self.emission_p[j,obs]
                backpointer_table[i, j] = np.argmax([viterbi_table[i-1][prev_state] * self.transition_p[prev_state][j] for prev_state in range(num_states)])

        # Step 3. Traceback 
        best_path_pointer = np.argmax(viterbi_table[-1])
        best_path = [best_path_pointer]
        for t in range(num_obs -1, 0, -1):
            best_path.insert(0, backpointer_table[t, best_path[0]])

        translated_best_path = [self.hidden_states_dict.get(index) for index in best_path]


        # Step 4. Return best hidden state sequence 
        return translated_best_path