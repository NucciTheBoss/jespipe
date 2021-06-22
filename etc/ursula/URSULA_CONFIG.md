# Configuring jespipe on Ursula

* [Install Python 3.9.5](#install-python-395)
* [Clone jespipe repository](#clone-jespipe-repository)
* [Set up Python 3.9.5 modulefile](#set-up-python-395-modulefile)
* [Install required packages](#install-required-packages)
* [Basic test to verify jespipe is ready to go](#basic-test-to-verify-jespipe-is-ready-to-go)
* [Commands to be run before each time you use jespipe](#commands-to-be-run-before-each-time-you-use-jespipe)

## Install Python 3.9.5

Use the following commands to install Python 3.9.5 from source on Ursula:

```tcsh
wget https://www.python.org/ftp/python/3.9.5/Python-3.9.5.tgz
tar -xzvf Python-3.9.5.tgz
cd Python-3.9.5
./configure --enable-shared --enable-optimizations --prefix=$HOME/sw/python-3.9.5
make && make install
cd .. && rm -rf Python-3.9.5 Python-3.9.5.tgz
```

Once Python 3.9.5 has completed its compilation and installation, use the following commands to set up the `python` and `pip` executables (by default `make && make install` only creates the `python3` and `pip3` executables):

```tcsh
cd ~/sw/python-3.9.5/bin
ln -s python3 python
ln -s pip3 pip
```

**IMPORTANT:** If you try to execute Python 3.9.5 right off the bat without any further environment configuration, you will probably receive an error message along the lines of `libpython3.9.so not found`. Do not worry as in the further steps your environment will be configured to use this locally installed version of Python 3.9.5.

## Clone jespipe repository

This one is pretty self explanitory. Use the following command to clone your own, local copy of the jespipe repository:

```tcsh
git clone https://github.com/NucciTheBoss/jespipe.git
```

## Set up Python 3.9.5 modulefile

To make your environment management easier, jespipe comes prepackaged with a TCL modulefile for Python 3.9.5. Use the following commands to set it up:

```tcsh
mkdir -p ~/sw/modules/python
cp jespipe/etc/ursula/3.9.5.tcl ~/sw/modules/python/3.9.5
```

**NOTE:** It is intentional that I left the `.tcl` file extension off on the copied file. This is because `Tmod` (the module system available on Ursula) does not need the `.tcl` file extension to process the TCL file. The `Tmod` documentation mandates that you leave off the extension.

## Install required packages

All the required Python modules are specified in the jespipe requirements.txt file in order to minimize the amount of scouring you need to do through PYPI. Simply use the following commands to install the required Python packages:

```tcsh
module use ~/sw/modules
module load mpi/openmpi-x86_64 python/3.9.5
pip install -r jespipe/requirements.txt
```

**IMPORTANT:** It is important that you load the `mpi/openmpi-x86_64` before attempting to install the packages mentioned in the requirements.txt file. This is because per Enterprise Linux and its derivatives (CentOS, Fedora, Oracle Linux, etc.) standards, `openmpi` is distributed as a TCL modulefile rather than a directly available executable. `openmpi` is a dependency for `mpi4py`, so `mpi4py` will fail to install if it cannot detect that `openmpi` as available

## Basic test to verify jespipe is ready to go

Tensorflow and Keras are critical component of jespipe, and while they might not be directly called by jespipe, it is crucial in many of its plug-and-play plugins. Therefore, it is important to verify that tensorflow is able to detect that `libcudart.so.11.0` is available on Ursula. Use the following Python import statement to verify that you can successfully import tensorflow and Keras:

```python
from tensorflow.keras.optimizers import Adam
```

If you see something like the message `successfully opened dynamic library libcudart.so.11.0` printed out, you should be good to go

## Commands to be run before each time you use jespipe

In order to successfully use jespipe for simulations on Ursula, you need to use the following commands before starting jespipe:

```tcsh
module use ~/sw/modules
module load mpi/openmpi-x86_64 python/3.9.5
```

Now you should be all set to start conducting simulations!
