Developer Guide
===============

This page provides information for developers who wish to contribute to
TOIREx or understand its internal structure.

Pipeline Architecture
---------------------

The pipeline consists of several independent modules responsible for
specific reduction tasks.

* Input and configuration handling
* Calibration
* Image processing
* Spectroscopic extraction
* Photometric extraction
* World Coordinate System calibration
* Output generation

The modules communicate primarily through FITS files and the pipeline
configuration, allowing individual reduction stages to be reused
independently.

Repository Structure
--------------------

The source code follows a standard Python package layout.

.. code-block:: text

   TOIREx/
   ├── docs/              # Documentation source
   ├── src/
   │   └── toirex/        # TOIREx source code
   ├── tests/             # Unit tests
   ├── pyproject.toml     # Package metadata
   ├── LICENSE
   └── README.md

The ``toirex`` package contains the complete implementation of the
pipeline.

Development Installation
------------------------

Clone the repository

.. code-block:: bash

   git clone https://github.com/varghesereji/TOIREx.git

Enter the repository

.. code-block:: bash

   cd TOIREx

Create a virtual environment

.. code-block:: bash

   python -m venv venv

Activate the environment

Linux/macOS

.. code-block:: bash

   source venv/bin/activate

Windows

.. code-block:: bat

   venv\Scripts\activate

Install TOIREx in editable mode

.. code-block:: bash

   pip install -e .

Editable installation allows modifications to the source code without
reinstalling the package.

Coding Style
------------

The TOIREx codebase follows the recommendations of :pep:`8`.

When contributing new code,

* use descriptive variable and function names,
* keep functions focused on a single task,
* avoid duplicated code,
* include informative comments where necessary,
* write docstrings for all public modules, classes and functions.

Docstrings should follow the NumPy documentation style.

Testing
-------

Before submitting any changes, ensure that the existing test suite passes.

Run all tests using

.. code-block:: bash

   pytest

If only a specific test file needs to be executed,

.. code-block:: bash

   pytest tests/test_filename.py

Documentation
-------------

The documentation is generated using Sphinx.

To build the documentation locally,

.. code-block:: bash

   cd docs
   make html

The generated documentation will be available in

.. code-block:: text

   docs/build/html/

Open ``index.html`` in a web browser to view the documentation.

Configuration File
------------------

Most pipeline behaviour is controlled through the configuration file.

When introducing a new configuration parameter,

* provide a sensible default value,
* document the parameter in the User Guide,
* ensure backward compatibility whenever possible,
* validate user input before use.

Logging
-------

TOIREx uses the Python ``logging`` module for status messages.

Developers should use

.. code-block:: python

   import logging

   logger = logging.getLogger(__name__)

instead of using ``print()`` statements for diagnostic output.

Error Handling
--------------

Errors should provide informative messages to help users identify the
problem quickly.

Where appropriate,

* validate inputs before processing,
* raise meaningful exceptions,
* avoid using bare ``except`` statements.

Contributing
------------

Contributions are welcome.

The recommended workflow is

1. Fork the repository.
2. Create a new feature branch.
3. Implement the changes.
4. Add or update tests where appropriate.
5. Update the documentation.
6. Submit a pull request.

Please ensure that all tests pass before opening a pull request.

API Documentation
-----------------

The API reference is generated automatically from the source code using
Sphinx.

When adding new public classes or functions,

* include complete docstrings,
* document parameters and return values,
* provide examples where appropriate.

Maintaining high-quality documentation helps both users and developers
understand and extend the TOIREx pipeline.
