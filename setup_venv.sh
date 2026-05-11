#!/bin/bash

cd ~/projects

python3 -m venv drl-routing-env
source drl-routing-env/bin/activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install numpy matplotlib networkx gymnasium pandas seaborn
pip install ns3gym

python3 -c "import torch; import numpy; import networkx; import gymnasium; import pandas; import seaborn; print('All packages imported successfully!')"
