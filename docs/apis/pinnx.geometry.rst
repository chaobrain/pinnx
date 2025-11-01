``pinnx.geometry`` module
=========================

.. currentmodule:: pinnx.geometry
.. automodule:: pinnx.geometry

This module provides geometric domains for defining the spatial and temporal extent of PDE problems.
Geometries support point sampling, boundary identification, and boolean operations for complex shapes.

Core Interfaces
---------------

Base classes and key geometric constructs.

**AbstractGeometry**: Base class for all geometries.

**Geometry**: Standard geometry interface.

**GeometryXTime**: Combines spatial geometry with time domain for time-dependent problems.

**TimeDomain**: Represents the temporal dimension.

**DictPointGeometry**: Geometry that works with dictionary-based points with physical units.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   AbstractGeometry
   Geometry
   GeometryXTime
   TimeDomain
   DictPointGeometry

Boolean Operations
------------------

Constructive solid geometry (CSG) operations for combining geometries.

**CSGUnion**: Union of two geometries (A ∪ B).

**CSGIntersection**: Intersection of two geometries (A ∩ B).

**CSGDifference**: Difference of two geometries (A \ B).

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   CSGUnion
   CSGIntersection
   CSGDifference

Canonical Domains
-----------------

Standard geometric shapes in 1D, 2D, 3D, and N-D.

**1D**: Interval

**2D**: Rectangle, Triangle, Polygon, Disk, Ellipse, StarShaped

**3D**: Cuboid, Sphere

**N-D**: Hypercube, Hypersphere

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   Interval
   Rectangle
   Triangle
   Polygon
   Disk
   Ellipse
   StarShaped
   Cuboid
   Sphere
   Hypercube
   Hypersphere

Point-Based Geometry
--------------------

Geometries defined by discrete point sets.

.. autosummary::
   :toctree: generated/
   :nosignatures:
   :template: classtemplate.rst

   PointCloud

