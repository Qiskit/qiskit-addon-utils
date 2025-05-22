######################
Qiskit addon utilities
######################

`Qiskit addons <https://quantum.cloud.ibm.com/docs/guides/addons>`_ are a collection of modular tools for building utility-scale workloads powered by Qiskit.

This package contains functionality which is meant to supplement workflows involving one or more Qiskit addons.
For example, this package contains functions for creating Hamiltonians, generating Trotter time evolution
circuits, and slicing and combining quantum circuits in time-wise partitions.

Documentation
-------------

All documentation is available `here <https://qiskit.github.io/qiskit-addon-utils/>`_.

Installation
------------

We encourage installing this package via ``pip``, when possible:

.. code-block:: bash

   pip install 'qiskit-addon-utils'


For more installation information refer to the `installation instructions <install.rst>`_ in the documentation.

Deprecation Policy
------------------

We follow `semantic versioning <https://semver.org/>`_ and are guided by the principles in
`Qiskit's deprecation policy <https://github.com/Qiskit/qiskit/blob/main/DEPRECATION.md>`_.
We may occasionally make breaking changes in order to improve the user experience.
When possible, we will keep old interfaces and mark them as deprecated, as long as they can co-exist with the
new ones.
Each substantial improvement, breaking change, or deprecation will be documented in the
`release notes <https://qiskit.github.io/qiskit-addon-utils/release-notes.html>`_.

Contributing
------------

The source code is available `on GitHub <https://github.com/Qiskit/qiskit-addon-utils>`_.

The developer guide is located at `CONTRIBUTING.md <https://github.com/Qiskit/qiskit-addon-utils/blob/main/CONTRIBUTING.md>`_
in the root of this project's repository.
By participating, you are expected to uphold Qiskit's `code of conduct <https://github.com/Qiskit/qiskit/blob/main/CODE_OF_CONDUCT.md>`_.

We use `GitHub issues <https://github.com/Qiskit/qiskit-addon-utils/issues/new/choose>`_ for tracking requests and bugs.

Citation
--------

If you use this package in your research, please cite it according to the `CITATION.bib <https://github.com/Qiskit/qiskit-addon-utils/blob/main/CITATION.bib>`_ file.

License
-------

`Apache License 2.0 <https://github.com/Qiskit/qiskit-addon-utils/blob/main/LICENSE.txt>`_


.. toctree::
  :hidden:

   Documentation Home <self>
   Installation Instructions <install>
   How-To Guides <how_tos/index>
   API Reference <apidocs/index>
   GitHub <https://github.com/qiskit/qiskit-addon-utils>
   Release Notes <release-notes>
