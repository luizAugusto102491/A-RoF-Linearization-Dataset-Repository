# Script to install the environment for SDR development and usage with GNU Radio
# Inatel CRR 2020
#
# Enable file to be executable after download it: $ sudo chmod +x ubuntu_18_sdr.sh
#

# Update and upgrade Ubuntu 18
sudo apt update
sudo apt upgrade -y

# Test for installed kernel version
MAJOR=`uname -r | awk -F "." '{print $1}'`;
MINOR=`uname -r | awk -F "." '{print $2}'`;
OLDKERNELS=`dpkg --list | grep linux-image | awk '{print $2}' | tr '\n' ' '`;

if [ "$MAJOR" -gt 4 ] || [ "$MINOR" -gt 15 ]
then
    echo "Incorrect Kernel version $MAJOR.$MINOR";
    echo "The kernel version must be lower than 4.18";
    echo "You you need to reboot the machine and run this script again"
    echo "Installing required kernel..."
    sudo apt install -y linux-headers-4.18.0-125-generic linux-image-4.18.0-125-generic linux-modules-4.18.0-125-generic linux-modules-extra-4.18.0-125-generic
    sudo sed -i 's/GRUB_DEFAULT=.*/GRUB_DEFAULT=\"Advanced options for Ubuntu>Ubuntu, with Linux 4.18.0-125-generic"/g' /etc/default/grub
    sudo update-grub
    echo ""
    echo ""
    echo "==============================================================================================="
    echo "Reboot the computer check the kernel version "\$uname -r" (4.18) then run this script again    "
    echo "==============================================================================================="
    echo ""
    echo ""
    echo "==============================================================================================="
    echo "Please, remove the newer kernel versions (5.4) present in your machine, because of National Intruments pcie driver!!!! Use the following commands"
    echo ""
    echo "sudo apt remove --purge linux-headers-5.4.* linux-image-5.4.* linux-modules-5.4.* linux-modules-extra-5.4.*"
    echo "sudo apt autoremove"
    echo "sudo update-grub"
    echo "==============================================================================================="
    echo ""
    echo ""
    exit 1;
fi

# Install development tools
sudo snap install code --classic
sudo apt install git vim -y
sudo apt install build-essential binutils cmake   -y # C/C++ 
sudo apt install python python-numpy python-scipy -y # Python stuff
sudo apt install python-pip -y
sudo apt install python3-pip -y
sudo apt install lm-sensors -y                       # System monitoring 

# Install GNU Radio environment
sudo apt install gnuradio gr-fosphor -y
sudo apt install swig -y
sudo apt install libboost-all-dev -y

# Volk Profile
sudo volk_profile


# Install libraries required for gr-inatel5g modem
sudo apt install libfftw3-* libuhd-dev -y
sudo apt install swig python-numba python-pyopencl -y
sudo apt install python-posix-ipc

# Change authentication method for proxy to solve git clone failing
git config --global http.proxyAuthMethod basic

# Create src directory and set permissions
mkdir -p ~/src
sudo chown $USER.$USER ~/src -R

# Install AFF3CT (Fast Forward Error Correction Toolbox)
cd ~/src
mkdir -p ~/src; cd ~/src
git clone --recursive https://github.com/aff3ct/aff3ct.git
cd ~/src/aff3ct
git checkout v2.3.5
mkdir build 
cd build
cmake .. -G"Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-funroll-loops -march=native" -DAFF3CT_COMPILE_EXE="ON" -DAFF3CT_COMPILE_STATIC_LIB="ON" -DAFF3CT_COMPILE_SHARED_LIB="ON"
git submodule update --init -- ../lib/rang/
cmake .. -G"Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-funroll-loops -march=native" -DAFF3CT_COMPILE_EXE="ON" -DAFF3CT_COMPILE_STATIC_LIB="ON" -DAFF3CT_COMPILE_SHARED_LIB="ON"
make -j10
sudo make install
sudo ldconfig
cd ~/


# ---------------------------------
# SDR drivers for XTRX SDR
# ---------------------------------
# Requisites
pip3 install cheetah3 # Reference: https://github.com/xtrx-sdr/images/issues/18

# XTRX USB Driver & libs
sudo apt install build-essential libusb-1.0-0-dev cmake dkms python-cheetah python -y
cd ~/
mkdir -p ~/src; cd ~/src
git clone --recursive https://github.com/xtrx-sdr/images.git
mv images xtrx-images
cd xtrx-images
git checkout 8c20ce1846eeebf4cf81e4025ed02003b6e71cda	# Commits on May 18, 2020 
cd sources
mkdir -p build ; cd build
cmake -DENABLE_SOAPY=NO -DINSTALL_UDEV_RULES=ON ..
make -j10
sudo make install
sudo ldconfig
sudo ln -s /usr/local/src/xtrx-0.0.1-2 /usr/src

# DKMS buils module
sudo /usr/sbin/dkms add -m xtrx -v "0.0.1-2"
sudo /usr/sbin/dkms build -m xtrx -v "0.0.1-2"
sudo /usr/sbin/dkms install -m xtrx -v "0.0.1-2"

# udev rules
cd ~/src/xtrx-images/sources
sudo cp xtrx_linux_pcie_drv/50-xtrx.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
sudo modprobe xtrx

# OsmoSDR (XTRX branch)
sudo apt install swig -y	# To include python support
cd ~/src
git clone https://github.com/xtrx-sdr/gr-osmosdr
cd gr-osmosdr/
git checkout 6e6ded6eb19c4445dbf28fd4ebd29faffe3e8acd	# Commits on Sep 27, 2019
mkdir build
cd build/
cmake ../
make -j10
sudo make install
sudo ldconfig


# ---------------------------------
# SDR drivers for Lime SDR
# ---------------------------------
# Dependencies
sudo apt install liboctave-dev libfltk1.3-dev libwxgtk-*-dev libsoapysdr-dev -y
cd ~/src
git clone https://github.com/myriadrf/LimeSuite.git
cd LimeSuite/
git checkout v19.04.0
mkdir builddir
cd builddir
cmake ../
make -j10
sudo make install
sudo ldconfig

# gr-limesdr
cd ~/src
git clone https://github.com/myriadrf/gr-limesdr
cd gr-limesdr/
git reset --hard  ccceb5c4b37aea58500c45b9bc22958f76ccce52 # HEAD Commit in November 4 2019 (tested) omit this line for newer versions
mkdir build
cd build/
cmake ..
make -j10
sudo make install
sudo ldconfig


# ---------------------------------
# SDR drivers for USRPs
# ---------------------------------
# USRP
sudo apt install libuhd003.010.003 libuhd-dev uhd-host -y
sudo apt install libosmosdr-dev qtbase5-dev libqcustomplot-dev -y
# Allow realtime priority (add current user to the usrp group)
sudo usermod -a -G usrp $USER 
sudo uhd_images_downloader
cd ~/src
wget https://codeload.github.com/EttusResearch/uhd/zip/release_003_010_003_000 -O uhd.zip
unzip uhd.zip
cd uhd-release_003_010_003_000/host/include
sudo cp -Rv uhd/rfnoc /usr/share/uhd/

# NI USP PCIe Driver
cd ~/src
wget http://files.ettus.com/binaries/niusrprio/niusrprio-installer-18.0.0.tar.gz
tar -xvzf niusrprio-installer-18.0.0.tar.gz
cd niusrprio_installer/
sudo ./INSTALL

# Starting NIUSRP Driver at boot time (script 01)
cd ~/src
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1p9uffXY9XidjSYeCsmpD-Oi2GKYTJgYx' -O niusrprio.service
sudo cp ~/src/niusrprio.service /etc/systemd/system/
sudo systemctl enable niusrprio.service
sudo systemctl start  niusrprio.service

# NI USRP PCIe Service (script 02 - review)
#cd ~/
#wget http://bit.do/niusrprio-service -O niusrprio.service		# !!!!!!!!!!!!!! Erro nessa linha !!!!!!!!!!!!!!
#sudo cp niusrprio.service /etc/systemd/system
#sudo systemctl daemon-reload
#sudo systemctl enable niusrprio.service
#sudo systemctl start  niusrprio.service


# ---------------------------------
# Other tools
# ---------------------------------
# Intel OpenCL Driver (newer boards Gen 7 onwards)
sudo add-apt-repository ppa:intel-opencl/intel-opencl -y
sudo apt update
sudo apt install intel-opencl-icd -y

# Install Fosphor (using gr-osmosdr from distribution)
#sudo apt install ocl-icd-* opencl-headers cmake xorg-dev libglu1-mesa-dev -y
#sudo apt install clinfo -y
#sudo apt install beignet -y
#sudo apt install qt4-default -y
#sudo apt install libglfw3-dev -y
#git clone git://git.osmocom.org/gr-fosphor
#cd gr-fosphor
#git checkout 7b6b9961bc2d9b84daeb42a5c8f8aeba293d207c
## Procurar pelo arquivo lib/fosphor/cl.c e no início da função cl_do_init comentar a linha self->flags |= FLG_FOSPHOR_USE_CLGL_SHARING;
#mkdir build
#cd build
#cmake -DCMAKE_INSTALL_PREFIX=/usr .
## Confirmar que qt e glfw estejam adicionados na compilação.
#make
#sudo make install
#sudo ldconfig

# IP Connectivity
sudo apt install net-tools bridge-utils uml-utilities iperf3 openssh-server -y

# Configuring iperf3 server at boot
cd ~/src
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=19t3EehRhwq0kwYSEH02S-55c-zyVgaaW' -O iperf3.service
sudo cp ~/src/iperf3.service /etc/systemd/system/
sudo systemctl enable iperf3.service
sudo systemctl start  iperf3.service

# Remote access suing Google Chrome Remote Desktop
sudo apt install xfce4-* -y


# ---------------------------------
# Create tun device
# ---------------------------------
#sudo tunctl -t radiopipe0 -g usrp # TUN interface
#sudo ip addr add 10.0.0.2/24 dev radiopipe0
#ip link set radiopipe0 mtu 9000
#sudo ip link set radiopipe0 up
#ip link set 


# ---------------------------------
# Find ethernet interface and configure Demo IP adresses
# !!! Used for demo wuth UC3M !!!
# ---------------------------------
#ETH=`nmcli dev status | grep ethernet  | awk '{ print $1 }'`
#OLDCON=`nmcli con show   | grep ethernet  | awk '{ print $1 }'`
#
#if [ "$1" == "phy1" ]
#then
#    echo "Configuring hostname $1 "
#    #hostnamectl set-hostname $1
#    echo "Configuring pipe interface $1 "
#    nmcli connection delete "$OLDCON"
#    nmcli connection add type tun mode tun group 27 ifname radiotun0 con-name radiotun0 eth.mtu 9000 ip4 10.154.253.138/30
#    nmcli connection add type ethernet con-name "phy1-eth" ifname $ETH ipv4.method manual ip4 10.154.253.130/30 # eth.mtu 9000
#    nmcli connection modify "radiotun0" +ipv4.routes  "10.154.253.132/30 10.154.253.137"
#    nmcli connection modify "radiotun0" +ipv4.routes  "10.4.0.0/16       10.154.253.137"
#    nmcli connection modify "phy1-eth"  +ipv4.routes  "10.154.253.0/26   10.154.253.129"
#    echo "Enable routing"
#    sudo sed -i 's/#net.ipv4.ip_forward=1/net.ipv4.ip_forward=1/g' /etc/sysctl.conf
#    sudo sysctl -w net.ipv4.ip_forward=1
#    sudo systemctl restart network-manager
#
#fi
#
#if [ "$1" == "phy2" ]
#then
#    echo "Configuring hostname $1 "
#    #hostnamectl set-hostname $1
#    echo "Configuring hostname $1 "
#    nmcli connection delete "$OLDCON"
#    nmcli connection add type tun mode tun group 27 ifname radiotun0 con-name radiotun0 eth.mtu 9000 ip4 10.154.253.137/30 
#    nmcli connection add type ethernet con-name "phy2-eth" ifname $ETH ipv4.method manual ip4 10.154.253.134/30 # eth.mtu 9000
#    nmcli connection modify "radiotun0" +ipv4.routes "10.154.253.128/30 10.154.253.138"
#    nmcli connection modify "radiotun0" +ipv4.routes "10.154.253.0/26   10.154.253.138"
#    nmcli connection modify "phy2-eth"  +ipv4.routes "10.4.0.0/16       10.154.253.133"
#    echo "Enable routing"
#    sudo sed -i 's/#net.ipv4.ip_forward=1/net.ipv4.ip_forward=1/g' /etc/sysctl.conf
#    sudo sysctl -w net.ipv4.ip_forward=1
#    sudo systemctl restart network-manager
#fi


#################################################################################################
#             Driver for ethernet pci for i9-13900K
###############################################################################################

#Model 8125 - Realtek Semiconductor Co.

# https://www.realtek.com/en/directly-download?downloadid=73865466490b208c00b7ea79734b7ac4













