#!/bin/bash
#SBATCH --job-name=pkg-check           # Job name
#SBATCH --partition=cpu-batch            # Use compute partition for CPU-only
#SBATCH --ntasks=1                     # One task
#SBATCH --cpus-per-task=1              # Just one CPU core needed
#SBATCH --mem=2G                       # Small memory allocation
#SBATCH --time=00:10:00                # Short runtime (10 minutes)
#SBATCH --output=pkg_versions_%j.out   # Output file
#SBATCH --error=pkg_versions_%j.err    # Error file

# Load system modules
source /etc/profile.d/modules.sh
module load python-ml/22-12-21-python3.9

# Check Python version
echo "Python version:"
python3.9 --version

# Check package versions
echo -e "\nPackage versions:"
python3.9 -c "
import sys
print(f'Python: {sys.version}')
try:
    import tensorflow as tf
    print(f'TensorFlow: {tf.__version__}')
except ImportError:
    print('TensorFlow: Not installed')
try:
    import keras
    try:
        print(f'Keras (standalone): {keras.__version__}')
    except:
        print(f'Keras: integrated with TensorFlow')
except ImportError:
    print('Keras: Not installed')
try:
    import numpy as np
    print(f'NumPy: {np.__version__}')
except ImportError:
    print('NumPy: Not installed')
try:
    import pandas as pd
    print(f'Pandas: {pd.__version__}')
except ImportError:
    print('Pandas: Not installed')
try:
    import matplotlib
    print(f'Matplotlib: {matplotlib.__version__}')
except ImportError:
    print('Matplotlib: Not installed')
try:
    import joblib
    print(f'Joblib: {joblib.__version__}')
except ImportError:
    print('Joblib: Not installed')
try:
    import h5py
    print(f'h5py: {h5py.__version__}')
except ImportError:
    print('h5py: Not installed')
"

# List all installed packages
echo -e "\nAll installed packages:"
python3.9 -m pip list

# Check if TensorFlow can access GPUs (for completeness)
echo -e "\nTensorFlow GPU availability (should be None/empty):"
python3.9 -c "
try:
    import tensorflow as tf
    print(f'GPU devices visible to TensorFlow: {tf.config.list_physical_devices(\"GPU\")}')
except Exception as e:
    print(f'Error checking TensorFlow GPU: {e}')
"

echo -e "\nComplete environment details saved to versions_full.txt"
python3.9 -c "import sys; print(sys.path)" > versions_full.txt
python3.9 -m pip freeze >> versions_full.txt