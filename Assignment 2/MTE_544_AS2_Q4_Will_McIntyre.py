"""
Student Name & Last Name: Will McIntyre
Origianl Author : Pi Thanacha Choopojcharoen
You must change the name of your file to MTE_544_AS2_Q4_(your full name).py
Do not use jupyter notebook.

*You may want to install the following libraries if you haven't done so.*

pip install numpy matplotlib pandas scipy

"""

import numpy as np

def decode_measurement(measurements):
    readings = []
    for measurement in measurements:
        readings.append(int(np.argmax(measurement) + 1))  # +1 for 1-based indexing
    return readings

def moodeng_behavior_update(state, A):
    # Given the current state, and the transition matrix A, 
    # randomly return the next state of Moo-Deng based on A

    # state is 1, 2, 3 so subtract 1 and get the column for this state
    transition_probs = A[:, state - 1]

    # randomly select the next state based on the weighted probabilities
    next_state = np.random.choice([1, 2, 3], p=transition_probs)
    return next_state

def sensor_measurement(state, C):
    # Given a state, and the matrix C, 
    # randomly return the encoded measurement based on C
    # Note that : 
    # F -> np.array([1,0,0])
    # R -> np.array([0,1,0])
    # P -> np.array([0,0,1])
    
    # state is 1, 2, 3 to subtract 1 and get the column for this state
    sensor_probs = C[:, state - 1]

    # randomly select the sensor reading based on weighted probs
    sensor_reading = np.random.choice([0,1,2], p=sensor_probs)

    # convert to the one-hot encoded array
    measurement = np.zeros(3)
    measurement[sensor_reading] = 1.0
    return measurement

def sim_moodeng(initial_state=1,iteration = 20):
    # Given an initial state of Moo-Deng's whereabout and number of iteration,
    # simulate Moo-Deng's behavior and the designed state estimator
    
    belief = np.array([1/3, 1/3, 1/3])  # Initial belief for the Bayesian filter
    
    A = np.array([[0.6, 0.2, 0.2], [0.4, 0.4, 0.1], [0, 0.4, 0.7]])
    C = np.array([[0.8, 0.2, 0.05], [0.1, 0.7, 0.1], [0.1, 0.1, 0.85]])

    states = []
    measurements = []
    estimated_states = []
    beliefs = []
    state = initial_state
    for i in range(iteration):
               
        # get next state and sensor measurement
        next_state = moodeng_behavior_update(state, A)
        measurement = sensor_measurement(next_state, C)

        # calculate prior (predicted x based on previous)
        prior = A @ belief # A * x_{k-1}

        # calculate likelihood, how measurement is affected by state
        likelihood = C.T @ measurement # C^T * y_k

        # calculate unnormalized belief and then the updated
        unnormalized_belief = likelihood * prior # element wise multiplication
        updated_belief = unnormalized_belief / np.sum(unnormalized_belief)

        # estimate the state by taking the state with highest prob
        estimated_state = np.argmax(updated_belief) + 1

        ##### END #####
        states.append(int(next_state))
        measurements.append(measurement)
        beliefs.append(updated_belief)
        estimated_states.append(int(estimated_state))

        state = next_state
        belief = updated_belief
        
    return states, measurements, estimated_states, beliefs

# Run the simulation
states, measurements, estimated, beliefs = sim_moodeng(initial_state=1, iteration=20)

readings = decode_measurement(measurements)
# Print results
print("True states:\n", states)
print("Sensor measurements:\n", readings)
print("Estimated states:\n", estimated)
print("Belief sequence:\n", np.array(beliefs))
