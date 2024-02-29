# A-RoF-Linearization-Dataset-Repository

This repository hosts a comprehensive dataset tailored for research in analog radio over fiber (A-RoF) systems, focusing particularly on linearization techniques. In A-RoF systems, mitigating nonlinear distortions is crucial, and while machine learning (ML)-based linearization schemes have been explored extensively, the lack of publicly available datasets hampers research progress. 

Here, we address this gap by offering a rich dataset. Leveraging a software-defined radio (SDR) approach, we generate digital samples of the generalized frequency division multiplexing (GFDM) waveform, which are then applied to a real A-RoF system. 

The dataset provided includes both transmitted and received GFDM waveform samples, enabling researchers effectively training ML models. By facilitating access to this dataset, we aim to foster the development of robust linearization schemes for A-RoF systems. Researchers can utilize this resource to enhance the effectiveness and robustness of ML-based linearization techniques, thus advancing the state-of-the-art in linearization A-RoF technology. 

*********************

## Summary :clipboard:
* [Dataset Information](#dataset-information)
* [How to Use](#how-to-use)
* [Example Usage](#example-usage)


*********************

 ## Dataset Information 📊 <a name="dataset-information"></a>

 ### Dataset Description:

The dataset provided here contains training instances and labels acquired from a practical A-RoF (Analog Radio-over-Fiber) system. Signals with bandwidths of 3, 6, 12, and 24 MHz were considered during data collection. For each bandwidth, RF power was varied from 0 to 11 dBm, resulting in eleven pairs of datasets for each bandwidth.

### File Format:

The files are saved in binary format, compatible with most Python IDEs.

### File Name Convention:

Each pair of datasets follows the naming convention:

   *   **Transmitted Samples:** (`dataset_n_rof_input_xdBm_yMHz`)
   *   **Received Samples:** (`dataset_n_rof_output_xdBm_yMHz`)

Where:

   * `n` varies from 1 to 11
   * `x` varies from 0 to 10
   * `y` represents the bandwidth, which can be 3, 6, 12 or 24 MHz

*********************

##  How to Use :arrow_forward: <a name="how-to-use"></a>

To utilize a dataset, follow these steps:

   * **Selecting Data Files:** Choose the transmitted file (`dataset_n_rof_input_xdBm_yMHz`) and its corresponding received file (`dataset_n_rof_output_xdBm_yMHz`).
  

   * **Loading Data:** Load the selected files into your Python environment using appropriate functions to read binary data.

   * **Processing:** Once loaded, process the data as required for your A-RoF linearization task.

### Example of How Load a Dataset in a Python IDE

After download the dataset from this repository, make sure to replace the 'path_to_tx_data' and 'path_to_rx_data' with the actual file paths on your system

```python

import numpy as np

# Replace 'path_to_tx_data' and 'path_to_rx_data' with the actual file paths on your system
tx_data_path = 'path_to_tx_data/dataset_1_rof_input_10dBm_6MHz'
rx_data_path = 'path_to_rx_data/dataset_1_rof_output_10dBm_6MHz'

# Load the transmission data from the file
tx_data = np.fromfile(tx_data_path, dtype=np.complex64)

# Load the received data from the file
rx_data = np.fromfile(rx_data_path, dtype=np.complex64)


```
*********************

##  Example Usage : <a name="example-usage"></a>

Below is a simple code snippet demonstrating how to load the dataset and train a basic neural network for A-RoF system linearization:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

# Load dataset
dataset = pd.read_csv('data/dataset.csv')
X = dataset.drop(columns=['output'])
y = dataset['output']

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train neural network model
model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000)
model.fit(X_train, y_train)

# Evaluate model
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

```
*********************



##  Setup and Network Configuration :white_check_mark: <a name="setup-installation"></a>

### Cloning the repo :file_folder:
First of all, to get a copy of the project, clone the repository into a folder named as `/home/<user_name>/src` on your machine:

```shell
git clone https://github.com/inatelcrr/inatel_xgrange_siso_p2p.git
```

The scripts folder has scripts for installation of all dependencies for this project and network configuration.

### Setup :computer:

**Considering you just installed Ubuntu 18.04 on your machine**. In `scripts/Setup` folder, there is a script called `ubuntu_18_sdr.sh`, which is responsible for configuring your machine and install all dependencies necessary for running this project. Run this script for the **first time** and it will verify if there is a newer kernel version. If there are kernel versions newer than `4.15`, please remove them after reboot for the first time and booting using kernel `4.15`:

``` shell
sudo apt remove --purge linux-headers-5.* linux-image-5.* linux-modules-5.* linux-modules-extra-5.*
sudo apt autoremove
```

After that, reboot your computer again and run the script again (**second time**). Doing that, all dependencies will be installed. So, finished all installation process, reboot the computer.

### Build project in GNU Radio :computer:

Entry in to `src/inatel_xgrange_siso_p2p` folder and execute the commands:

``` shell
mkdir build
cd build
cmake ../
make -j5
sudo make install
sudo ldconfig
```

### Network configuration  :computer:

The project can be used as two kind of terminal: **base station (BS)** and **user equipment (UE)**. Each one has a specific script for network configuration inside `scripts/Network` folder, because is used a `tun` kernel virtual network device for doing routing process and communicate with phy layer.
*   **Configuring BS**: For the computer that will run as BS, please run the script `config_network_bs.sh`.
*   **Configuring UE**: For the computer that will run as UE, please run the script `config_network_ue.sh`.
