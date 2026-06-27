API Reference
=============

demtools provides the following public classes.  All classes live in
``demtools.grids`` (or ``demtools.storage`` for :class:`~demtools.storage.H5Store`)
and are re-exported from the top-level ``demtools`` package.

Each class documents only its **own** methods and attributes.  Because the full
class hierarchy is documented here, inherited methods appear in the section for
the class that defines them — use the **Show inheritance** banner to navigate
between levels.

Grid (base class)
-----------------

.. autoclass:: demtools.grids.Grid
    :members:
    :show-inheritance:

BoolGrid
--------

.. autoclass:: demtools.grids.BoolGrid
    :members:
    :show-inheritance:

IntGrid
-------

.. autoclass:: demtools.grids.IntGrid
    :members:
    :show-inheritance:

FloatGrid
---------

.. autoclass:: demtools.grids.FloatGrid
    :members:
    :show-inheritance:

DEMGrid
-------

.. autoclass:: demtools.grids.DEMGrid
    :members:
    :show-inheritance:

RGBimage
--------

.. autoclass:: demtools.grids.RGBimage
    :members:
    :show-inheritance:

FeatureSet
----------

.. autoclass:: demtools.grids.FeatureSet
    :members:
    :show-inheritance:

H5Store
-------

.. autoclass:: demtools.storage.H5Store
    :members:
    :show-inheritance:
