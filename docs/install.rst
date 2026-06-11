Install
=========================

There are two primary ways to run and install the
packages:

- :ref:`Option 1`
- :ref:`Option 2`

Preinstallation
^^^^^^^^^^^^^^^^

Create a minimal environment with only Python installed in it. We recommend using `Python virtual environments <https://docs.python.org/3.10/tutorial/venv.html>`__.

.. code:: sh

    python3 -m venv /path/to/virtual/environment

Activate your new environment.

.. code:: sh

    source /path/to/virtual/environment/bin/activate

Note: If you are using Windows, run the following commands in PowerShell:

.. code:: pwsh

    python3 -m venv c:\path\to\virtual\environment
    c:\path\to\virtual\environment\Scripts\Activate.ps1


.. _Option 1:

Option 1: Install from PyPI
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The most straightforward way to install the ``qiskit-addon-utils`` package is by using ``PyPI``.

.. code:: sh

    pip install 'qiskit-addon-utils'


.. _Option 2:

Option 2: Install from source
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Users who want to develop in the repository or run the notebooks locally might want to install from source.

If so, the first step is to clone the ``qiskit-addon-utils`` repository.

.. code:: sh

    git clone git@github.com:Qiskit/qiskit-addon-utils.git

Next, upgrade pip and enter the repository.

.. code:: sh

    pip install --upgrade pip
    cd qiskit-addon-utils

The next step is to install ``qiskit-addon-utils`` to the virtual environment. If you plan on running the notebooks, install the
notebook dependencies. If you plan on developing in the repository, you
might want to install the ``dev`` dependencies.

Adjust the options below to suit your needs.

.. code:: sh

    pip install tox notebook -e '.[notebook-dependencies,dev]'

If you installed the notebook dependencies, you can get started by running the notebooks in the docs.

.. code::

    cd docs/
    jupyter lab
