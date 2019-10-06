.. ###########################################################################
.. README.rst
.. ==========
.. Author : Sepand KASHANI [sepand.kashani@epfl.ch]
.. ###########################################################################


########
DeepWave
########

This repository contains the reference implementation of `DeepWave
<https://infoscience.epfl.ch/record/265765?ln=en>`_.


Installation
------------

After installing `Miniconda <https://conda.io/miniconda.html>`_ or `Anaconda
<https://www.anaconda.com/download/#linux>`_, run the following::

    $ conda create --name=DeepWave python=3.6
    $ source activate DeepWave

    # Install `ImoT_tools <https://github.com/imagingofthings/ImoT_tools>`_
    $ git clone git@github.com:imagingofthings/ImoT_tools.git
    $ cd ImoT_tools
    $ conda install --channel=defaults --channel=conda-forge --file=requirements.txt
    $ python3 setup.py develop
    $ cd ..

    # Install `DeepWave <https://github.com/imagingofthings/DeepWave>`_
    $ git clone git@github.com:imagingofthings/DeepWave.git
    $ cd DeepWave
    $ python3 setup.py develop
