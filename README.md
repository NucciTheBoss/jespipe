<p align="center">
    <i>Image for branding?</i>
</p>

<h3 align="center">Jespipe</h3>

<p align="center">
    <i>Short description about Jespipe</i>
    <br />
    <i>Invitation to explore the Jespipe wiki pages</i>
    <br />
    <i>Important links for the users</i>
</p>

## Jespipe v0 - Experimental

## Table of Contents

* [Quick Start](#quick-start)
* [What's Included](#whats-included)
* [Documentation](#documentation)
* [Bugs and Feature Requests](#bugs-and-feature-requests)
* [Contributing](#contributing)
* [Creators](#creators)
* [Copyright and License](#copyright-and-license)

## Quick Start
* [System Requirements](#system-requirements)
* [Install Dependencies](#install-dependencies)
* [Install OpenMPI](#install-openmpi)
* [Install Jespipe](#install-jespipe)

### System Requirements

The recommended runtime environment for Jespipe is on High-Performance Computing clusters, but your system only has to meet the following minimum requirements in order to run Jespipe:

| Category | Minimum Requirement |
| :--- | :---: |
| **Operating System:** | CentOS 7 or newer, Debian 10 or newer, OpenSUSE 15.3 or newer, RedHat Enterprise Linux 7 or newer, Ubuntu 18.04 or newer |
| **Number of Processor Cores:** | Bare minimum: 2 cores - 4 threads; Recommended: 8 cores - 16 threads |
| **RAM:** | Bare minimum: 4GB; Recommended: 16GB |
| **GPU:** | *Not required but recommended for larger-scale models and computationally instensive attacks.* |
| **Python Version:** | 3.5+, 3.8+, 3.9.5, or 3.9.6 |

**It is strongly recommended** that you use Jespipe on a High-Performance Computing cluster because while Jespipe's system resource consumption is minimal, under-powered machines will be quickly overwhelmed by large-scale adversarial analyses.

### Install Dependencies

Use the following commands to install Jespipe's system-level dependencies according to your operating system.

#### CentOS / RedHat Enterprise Linux:

```bash
sudo yum update
sudo yum groupinstall "Development Tools" "Development Libraries"
wget https://www.python.org/ftp/python/3.9.6/Python-3.9.6.tar.xz -O - | tar -xJv
cd Python-3.9.6
make && sudo make install
```

#### OpenSUSE:

```bash
sudo zypper update
sudo zypper install -t pattern devel_basis
sudo zypper install python39-base python39-devel python39-pip
```

#### Ubuntu / Debian:

```bash
sudo apt-get update
sudo apt-get install build-essential python3.9-full python3.9-dev python3-pip
```

### Install OpenMPI

Use the following commands to install OpenMPI according to your operating system.

#### CentOS / RedHat Enterprise Linux:

```bash
sudo yum install openmpi openmpi-devel
```

#### OpenSUSE:

```bash
sudo zypper install openmpi4 openmpi4-libs openmpi4-devel openmpi4-config openmpi4-docs
```

#### Ubuntu / Debian:

```bash
sudo apt-get install openmpi-bin openmpi-common openmpi-doc
```

### Install Jespipe

With the system-level dependencies and OpenMPI installed on your system, use the following commands to set Jespipe up on your system.

#### Install from release (for production):

To install Jespipe from a release, use the following commands:

```bash
wget <jespipe-release-goes-here> -O - | tar -xzv -
cd jespipe-<release-number>
pip install -r requirements.txt
python setup.py install
```

#### Install from source (for development):

To install Jespipe from source, use the following commands:

```bash
git clone https://github.com/NucciTheBoss/jespipe.git
cd jespipe
pip install -r requirements.txt
python setup.py install
```

#### Test Installation:

If all went well, you should be able to print out Jespipe's version info without issue:

```
$ python main.py --version

       _                 _                     ___   ___  __ 
      | |               (_)                   / _ \ / _ \/_ |
      | | ___  ___ _ __  _ _ __   ___  __   _| | | | | | || |
  _   | |/ _ \/ __| '_ \| | '_ \ / _ \ \ \ / / | | | | | || |
 | |__| |  __/\__ \ |_) | | |_) |  __/  \ V /| |_| | |_| || |
  \____/ \___||___/ .__/|_| .__/ \___|   \_/  \___(_)___(_)_|
                  | |     | |                                
                  |_|     |_|                                
 

Jespipe-v0.0.1  Copyright (C) 2021  Jason C. Nucciarone, Eric Inae, Sheila Alemany

This program comes with ABSOLUTELY NO WARRANTY; for details please see the LICENSE file. 
This is free software, and you are welcome to redistribute it
under certain conditions; please visit https://github.com/NucciTheBoss/jespipe for more details.
```

## What's Included
```
File tree showing avaiable files
```
*Fast overview of all the directories included in the source release*.

## Documentation
*Explain how to access documentation*.

*Hyperlink to direct to the wiki pages*.

*How to access documentation files include in the source release of Jespipe*.

## Bugs and Feature Requests
*Explain how to open an issue*.

*Explain how to open a feature request*.

## Contributing
*Hyperlink to CONTRIBUTING.md file*.

## Creators
* Jason C. Nucciarone - Pennsylvania State University - Primary Author
  * ~~[Jason's Personal Site](https://nucci.tech)~~
  * [LinkedIn](https://www.linkedin.com/in/jasonnucci/)

* Eric Inae - Andrews University - Co-Author
  * [LinkedIn](https://www.linkedin.com/in/eric-inae-6056b1214)

* Sheila Alemany - Florida International University - Co-Author
  * [Sheila's Personal Site](https://sheilaalemany.github.io/)
  * [LinkedIn](https://www.linkedin.com/in/sheilaalemany)

## Copyright and License
Code and documentation copyright &copy; 2021 Jason C. Nucciarone, Eric Inae, and Sheila Alemany. Code released under the [GNU General Public License version 3](https://www.gnu.org/licenses/gpl-3.0.en.html). Documentation is released under the [BSD 3-Clause License](https://opensource.org/licenses/BSD-3-Clause).
