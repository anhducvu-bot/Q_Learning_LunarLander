# DQN agent that learn how to land the Lunar Lander on the Moon through trials and errors!

# What? 

1. Environment: 
The RL agent will learn in the environment Lunar Lander by Open AI. Below is a small description of the environment: 

"Landing pad is always at coordinates (0,0). Coordinates are the first two numbers in state vector. Reward for moving from the top of the screen to landing pad and zero speed is about 100..140 points. If lander moves away from landing pad it loses reward back. Episode finishes if the lander crashes or comes to rest, receiving additional -100 or +100 points. Each leg ground contact is +10. Firing main engine is -0.3 points each frame. Solved is 200 points. Landing outside landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land on its first attempt. Four discrete actions available: do nothing, fire left orientation engine, fire main engine, fire right orientation engine." 

More information about the environment can be found [here](https://gym.openai.com/envs/LunarLander-v2/)

2. The AI: 
The agent is a DQN agent that use cyclical learning rate. 

# How? 

The users need to import package gym to run the algorithm. 

# Performance:

Before Trainining: [here](https://youtu.be/j1jusWqi4eA/)

After Training: [Click Here] (https://www.youtube.com/watch?v=q24c_dQowF0/) 


