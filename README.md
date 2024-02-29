# A-RoF-Linearization-Dataset-Repository

This repository hosts a comprehensive dataset tailored for research in analog radio over fiber (A-RoF) systems, focusing particularly on linearization techniques. In A-RoF systems, mitigating nonlinear distortions is crucial, and while machine learning (ML)-based linearization schemes have been explored extensively, the lack of publicly available datasets hampers research progress. 

Here, we address this gap by offering a rich dataset. Leveraging a software-defined radio (SDR) approach, we generate digital samples of the generalized frequency division multiplexing (GFDM) waveform, which are then applied to a real A-RoF system. 

The dataset provided includes both transmitted and received GFDM waveform samples, enabling researchers to train ML models effectively. By facilitating access to this dataset, we aim to foster the development of robust linearization schemes for A-RoF systems. Researchers can utilize this resource to enhance the effectiveness and robustness of ML-based linearization techniques, thus advancing the state-of-the-art in A-RoF technology. 


## Summary :clipboard:

* [Project Description](#project-description)
* [How to Use](#how-to-use)
* [Dataset Information](#dataset-information)
* [Example Usage](#example-usage)

*********************

##  Project Description :rocket: <a name="project-description"></a>

*********************

 ## How To Use :arrow_forward: <a name="how-to-use"></a>

The `examples` folder contains `GRC` examples for transmitter usage. We have two versions of GRC in order to test this project in two ways. 
*   **Virtual Loopback:** The first one is the project inside `examples/virtual` folder, called `virtual_xG_Range_loopback.grc`. It runs without using an `USRP` device, that is, it is possible to run the project just in virtual loopback mode to validate the communication. 
*   **Using SDR/RF (NI USRP):** The second one is the project inside `examples/sdr` folder, called `modem_xG_Range_siso_pp.grc`. It runs using an `USRP` device and it was tested with **NI USRP-2954** and **NI USRP-2952**. In order to communicate **BS** and **UE**, just change the frequency configuration inside the flowgraph that satisfies the following constaints:
    *   The `downlink_freq` in BS flowgraph should be equal to `uplink_freq` in UE;
    *   The `uplink_freq` in BS flowgraph should be equal to `downlink_freq` in UE;

    **It is possible to run the communication just using one computer, doing a kind of Loopback but using RF/SDR. For that, the `downlink_freq` should be equal to `uplink_freq`.**

### Testing  :computer:

In order to test the communication between BS and UE terminals when using `RF/SDR (NI USRP)`, it is possible running the following commands:

In BS (to ping UE):

``` shell
ping 10.0.0.2
```

Or In UE (to ping BS):

``` shell
ping 10.0.0.1
```

*********************

##  Dataset Information 📊: <a name="dataset-information"></a>


*********************

##  Example Usage💻: <a name="example-usage"></a>

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
