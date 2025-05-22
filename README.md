<!-- SHIELDS -->
<div align="left">

  [![Release](https://img.shields.io/pypi/v/qiskit-addon-utils.svg?label=Release)](https://github.com/Qiskit/qiskit-addon-utils/releases)
  ![Platform](https://img.shields.io/badge/%F0%9F%92%BB%20Platform-Linux%20%7C%20macOS%20%7C%20Windows-informational)
  [![Python](https://img.shields.io/pypi/pyversions/qiskit-addon-utils?label=Python&logo=python)](https://www.python.org/)
  [![Qiskit](https://img.shields.io/badge/Qiskit%20-%20%3E%3D1.2%20-%20%236133BD?logo=Qiskit)](https://github.com/Qiskit/qiskit)
<br />
  [![Docs (stable)](https://img.shields.io/badge/%F0%9F%93%84%20Docs-stable-blue.svg)](https://qiskit.github.io/qiskit-addon-utils/)
  [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13711854.svg)](https://zenodo.org/doi/10.5281/zenodo.13711854)
  [![License](https://img.shields.io/github/license/Qiskit/qiskit-addon-utils?label=License)](LICENSE.txt)
  [![Downloads](https://img.shields.io/pypi/dm/qiskit-addon-utils.svg?label=Downloads)](https://pypi.org/project/qiskit-addon-utils/)
  [![Tests](https://github.com/Qiskit/qiskit-addon-utils/actions/workflows/test_latest_versions.yml/badge.svg)](https://github.com/Qiskit/qiskit-addon-utils/actions/workflows/test_latest_versions.yml)
  [![Coverage](https://coveralls.io/repos/github/Qiskit/qiskit-addon-utils/badge.svg?branch=main)](https://coveralls.io/github/Qiskit/qiskit-addon-utils?branch=main)

# Qiskit addon utilities

### Table of contents

* [About](#about)
* [Documentation](#documentation)
* [Installation](#installation)
* [Deprecation Policy](#deprecation-policy)
* [Contributing](#contributing)
* [Citation](#citation)
* [License](#license)

----------------------------------------------------------------------------------------------------

### About

[Qiskit addons](https://quantum.cloud.ibm.com/docs/guides/addons) are a collection of modular tools for building utility-scale workloads powered by Qiskit.

This package contains functionality which is meant to supplement workflows involving one or more Qiskit addons.
For example, this package contains functions for creating Hamiltonians, generating Trotter time evolution
circuits, and slicing and combining quantum circuits in time-wise partitions.

----------------------------------------------------------------------------------------------------

### Documentation

All documentation is available at https://qiskit.github.io/qiskit-addon-utils/.

----------------------------------------------------------------------------------------------------

### Installation

We encourage installing this package via `pip`, when possible:

```bash
pip install 'qiskit-addon-utils'
```

For more installation information refer to these [installation instructions](docs/install.rst).

----------------------------------------------------------------------------------------------------

### Deprecation Policy

We follow [semantic versioning](https://semver.org/) and are guided by the principles in
[Qiskit's deprecation policy](https://github.com/Qiskit/qiskit/blob/main/DEPRECATION.md).
We may occasionally make breaking changes in order to improve the user experience.
When possible, we will keep old interfaces and mark them as deprecated, as long as they can co-exist with the
new ones.
Each substantial improvement, breaking change, or deprecation will be documented in the
[release notes](https://qiskit.github.io/qiskit-addon-utils/release-notes.html).

----------------------------------------------------------------------------------------------------

### Contributing

The source code is available [on GitHub](https://github.com/Qiskit/qiskit-addon-utils).

The developer guide is located at [CONTRIBUTING.md](https://github.com/Qiskit/qiskit-addon-utils/blob/main/CONTRIBUTING.md)
in the root of this project's repository.
By participating, you are expected to uphold Qiskit's [code of conduct](https://github.com/Qiskit/qiskit/blob/main/CODE_OF_CONDUCT.md).

We use [GitHub issues](https://github.com/Qiskit/qiskit-addon-utils/issues/new/choose) for tracking requests and bugs.

----------------------------------------------------------------------------------------------------

### Citation

If you use this package in your research, please cite it according to the [CITATION.bib](https://github.com/Qiskit/qiskit-addon-utils/blob/main/CITATION.bib) file.

----------------------------------------------------------------------------------------------------

### License

[Apache License 2.0](LICENSE.txt)
