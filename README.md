# NeuralOperator2x2Hyperbolic
The source code for the paper is named Backstepping Neural Operators for $2\times 2$ Hyperbolic PDEs. All the code is written in Python and relies on standard packages such as numpy, Pytorch, and the deep learning package DeepXDE.

# Step 1： Learning kernels
Please see the file in the folder named learn_kernels.py. This model can used to train all kernels. 

# Step 2： Learning output feedback law
Please see the file in the folder named learn_U.py. This model can used to train output feedback law. 

# Step 3： Simulation examples
Please see the file in the folder named Hyperbolic_system.py. This file can generate the dataset for Steps 1 and 2, and also can use train models for two simulation examples of 2X2 Hyperbolic PDE. 

# Questions
Feel free to leave any questions in the issues of Github or email at wss_dhu@126.com.
