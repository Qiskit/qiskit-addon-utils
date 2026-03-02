# This code is a Qiskit project.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import inspect
import os
import re
import sys
from importlib.metadata import version as metadata_version

# The following line is required for autodoc to be able to find and import the code whose API should
# be documented.
sys.path.insert(0, os.path.abspath(".."))

project = "Qiskit addon utilities"
project_copyright = "2024, Qiskit addons team"
description = "Utilities to support workflows leveraging Qiskit addons"
author = "Qiskit addons team"
language = "en"
release = metadata_version("qiskit-addon-utils")

html_theme = "qiskit-ecosystem"

# This allows including custom CSS and HTML templates.
html_theme_options = {
    "dark_logo": "images/qiskit-dark-logo.svg",
    "light_logo": "images/qiskit-light-logo.svg",
    "sidebar_qiskit_ecosystem_member": False,
}
html_static_path = ["_static"]
templates_path = ["_templates"]

# Sphinx should ignore these patterns when building.
exclude_patterns = [
    "_build",
    "_ecosystem_build",
    "_qiskit_build",
    "_pytorch_build",
    "**.ipynb_checkpoints",
    "jupyter_execute",
]

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.mathjax",
    "sphinx.ext.linkcode",
    "sphinx.ext.intersphinx",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_copybutton",
    "sphinx_reredirects",
    "reno.sphinxext",
    "nbsphinx",
    "qiskit_sphinx_theme",
    "pytest_doctestplus.sphinx.doctestplus",
]

html_last_updated_fmt = "%Y/%m/%d"
html_title = f"{project} {release}"

# This allows RST files to put `|version|` in their file and
# have it updated with the release set in conf.py.
rst_prolog = f"""
.. |version| replace:: {release}
"""

# Options for autodoc. These reflect the values from Qiskit SDK and Runtime.
autosummary_generate = True
autosummary_generate_overwrite = False
autoclass_content = "both"
autodoc_typehints = "description"
autodoc_default_options = {
    "inherited-members": None,
    "show-inheritance": True,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = False


# This adds numbers to the captions for figures, tables,
# and code blocks.
numfig = True
numfig_format = {"table": "Table %s"}

# Settings for Jupyter notebooks.
nbsphinx_execute = "never"

add_module_names = False

modindex_common_prefix = ["qiskit_addon_utils."]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "qiskit": ("https://quantum.cloud.ibm.com/docs/api/qiskit/", None),
    "rustworkx": ("https://www.rustworkx.org/", None),
}

plot_working_directory = "."
plot_html_show_source_link = False

# ----------------------------------------------------------------------------------
# Redirects
# ----------------------------------------------------------------------------------

_inlined_apis = [
    ("qiskit_addon_utils.coloring", "auto_color_edges"),
    ("qiskit_addon_utils.coloring", "is_valid_edge_coloring"),
    ("qiskit_addon_utils.problem_generators", "generate_xyz_hamiltonian"),
    ("qiskit_addon_utils.problem_generators", "generate_time_evolution_circuit"),
    ("qiskit_addon_utils.problem_generators", "PauliOrderStrategy"),
    ("qiskit_addon_utils.slicing", "combine_slices"),
    ("qiskit_addon_utils.slicing", "combine_slices"),
    ("qiskit_addon_utils.slicing", "slice_by_barriers"),
    ("qiskit_addon_utils.slicing", "slice_by_coloring"),
    ("qiskit_addon_utils.slicing", "slice_by_depth"),
    ("qiskit_addon_utils.slicing", "slice_by_gate_types"),
    ("qiskit_addon_utils.exp_vals", "executor_expectation_values"),
    ("qiskit_addon_utils.exp_vals", "get_measurement_bases"),
    ("qiskit_addon_utils.exp_vals", "map_observable_isa_to_canonical"),
    ("qiskit_addon_utils.exp_vals", "map_observable_virtual_to_canonical"),
    ("qiskit_addon_utils.exp_vals", "map_observable_isa_to_virtual"),
    ("qiskit_addon_utils.noise_management", "PostSelectionSummary"),
    ("qiskit_addon_utils.noise_management", "PostSelector"),
    ("qiskit_addon_utils.noise_management", "gamma_from_noisy_boxes"),
    ("qiskit_addon_utils.noise_management", "trex_factors"),
]

redirects = {
    "apidocs/qiskit_addon_utils": "./index.html",
    "apidocs/qiskit_addon_utils.transpiler": "./index.html",
    **{
        f"stubs/{module}.{name}": f"../apidocs/{module}.html#{module}.{name}"
        for module, name in _inlined_apis
    },
}

# ----------------------------------------------------------------------------------
# Source code links
# ----------------------------------------------------------------------------------


def determine_github_branch() -> str:
    """Determine the GitHub branch name to use for source code links.

    We need to decide whether to use `stable/<version>` vs. `main` for dev builds.
    Refer to https://docs.github.com/en/actions/learn-github-actions/variables
    for how we determine this with GitHub Actions.
    """
    # If CI env vars not set, default to `main`. This is relevant for local builds.
    if "GITHUB_REF_NAME" not in os.environ:
        return "main"

    # PR workflows set the branch they're merging into.
    if base_ref := os.environ.get("GITHUB_BASE_REF"):
        return base_ref

    ref_name = os.environ["GITHUB_REF_NAME"]

    # Check if the ref_name is a tag like `1.0.0` or `1.0.0rc1`. If so, we need
    # to transform it to a Git branch like `stable/1.0`.
    version_without_patch = re.match(r"(\d+\.\d+)", ref_name)
    return f"stable/{version_without_patch.group()}" if version_without_patch else ref_name


GITHUB_BRANCH = determine_github_branch()


def linkcode_resolve(domain, info):
    """Add links to GitHub source code."""
    if domain != "py":
        return None

    module_name = info["module"]
    module = sys.modules.get(module_name)
    if module is None or "qiskit_addon_utils" not in module_name:
        return None

    def is_valid_code_object(obj):
        return inspect.isclass(obj) or inspect.ismethod(obj) or inspect.isfunction(obj)

    obj = module
    for part in info["fullname"].split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None
        if not is_valid_code_object(obj):
            return None

    # Unwrap decorators. This requires they used `functools.wrap()`.
    while hasattr(obj, "__wrapped__"):
        obj = obj.__wrapped__
        if not is_valid_code_object(obj):
            return None

    try:
        full_file_name = inspect.getsourcefile(obj)
    except TypeError:
        return None
    if full_file_name is None or "/qiskit_addon_utils/" not in full_file_name:
        return None
    file_name = full_file_name.split("/qiskit_addon_utils/")[-1]

    try:
        source, lineno = inspect.getsourcelines(obj)
    except (OSError, TypeError):
        linespec = ""
    else:
        ending_lineno = lineno + len(source) - 1
        linespec = f"#L{lineno}-L{ending_lineno}"
    return f"https://github.com/Qiskit/qiskit-addon-utils/tree/{GITHUB_BRANCH}/qiskit_addon_utils/{file_name}{linespec}"
