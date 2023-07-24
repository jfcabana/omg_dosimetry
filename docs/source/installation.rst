.. _installation:

============
Installation
============

Installing OMG Dosimetry is easy no matter your python skills! Determine where you are at 
and then read the relevant section:

I know Python
=============

Python best practices recommend creating separate virtual environments for your different applications,
to avoid dependency version conflicts. The following commands will install a dedicated Python environment for OMG Dosimetry using conda. 

.. code-block:: bash

    $ conda create --name omg_d python=3.10
    $ pip install omg-dosimetry

.. _dependencies:

Dependencies
------------

OMG Dosimetry, as a scientific package, has fairly standard scientific dependencies (>= means at least that version or newer).
Installing the package via ``pip`` will install these for you:

.. literalinclude:: ../../requirements_pip.txt

I'm new to Python
=================

That's okay! If you're not a programmer at all you'll have a few things to do to get up. Using OMG Dosimetry requires not just the base language Python, but a few dependencies as well.
Since most physicists don't program, or if they do it's in MATLAB, this section will help jumpstart your use of not
just OMG Dosimetry but Python in general and all its wonderful goodness! Getting started with Python takes some work to
get set up, but it's well worth the effort.

.. _distro_stack:

Get a Distribution Stack
------------------------

Scientific computing with Python requires some specialized packages which require some specialized computing libraries.
While it's possible you have those libraries (for some odd reason), it's not likely. Thus, it's often best to install
the libraries *pre-compiled*. There are several options out there; I'll list just a few. Be sure to download the 3.x version,
preferably the newest:

* `Anaconda <https://www.anaconda.com/download>`_ provides this one-stop-shop for tons of
  scientific libraries in an easy to install format. Just download and run the installer. If you don't want to install
  all 200+ packages, a slimmer option exists: `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_, which only installs
  ``conda`` and python installation tools.
  Here's the Anaconda `Start guide <https://docs.anaconda.com/free/anaconda/#installation>`_.

Install OMG Dosimetry
---------------------

After Anaconda has been installed, open Anaconda Prompt. Inside it (a window with black background), run the following commands:

.. code-block:: bash

    $ conda create --name omg_d python=3.10
    $ pip install omg-dosimetry

Get an IDE
^^^^^^^^^^

* `Spyder <https://www.spyder-ide.org/>`_ - A MATLAB-like IDE with similar layout, preferred by many working in the scientific realm.
  Here are the `Spyder docs <https://docs.spyder-ide.org/current/index.html>`_.

  .. note:: Spyder is part of the Anaconda distribution.

  .. image:: https://docs.spyder-ide.org/current/_images/mainwindow_default_1610.png
     :height: 400px
     :width: 600px

Inspiration
^^^^^^^^^^^

This installation guide is heavily inspired by `GitHub's pylinac instructions <https://pylinac.readthedocs.io/en/latest/installation.html>`_.
