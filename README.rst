.. ###########################################################################
.. README.rst
.. ==========
.. Author : Sepand KASHANI [sepand.kashani@epfl.ch]
.. ###########################################################################


###############
Acoustic Camera
###############

This repository contains the working code for "See What You Hear: A Super-
Resolved, Neural Network-Powered, Real-Time Acoustic Camera [Sepand KASHANI,
Matthieu SIMEONI".


Installation
------------
conda create --name=acoustic_camera \
             --channel=defaults \
             --channel=conda-forge \
             --file=conda_requirements.txt
conda activate acoustic_camera
python3 setup.py develop
