Configuration and Setup
=============

TOIREx is controlled through an INI-style configuration file. The
configuration file is divided into several sections, each controlling a
different stage of the reduction pipeline.

Setting Up
----------

Download the default configuration file from

https://github.com/varghesereji/TOIREx/blob/main/src/toirex/config/TOIREX_config.config

or copy it from your TOIREx installation.

.. code-block:: bash

   /path/to/env/TOIREx/src/toirex/config/TOIREX_config.config

Place the configuration file in the same directory as your observational
data. Modify the required parameters before running the pipeline.

Configuration Sections
----------------------

The configuration file contains the following sections:

* ``[inits]`` – General pipeline settings.
* ``[outputs]`` – Output directory and filenames.
* ``[visual]`` – Display options.
* ``[logging]`` – Logging configuration.
* ``[inputs]`` – Input image processing.
* ``[dither]`` – Dither alignment.
* ``[spectral_extraction]`` – Spectroscopic extraction.
* ``[wcs]`` – World Coordinate System parameters.
* ``[photometry]`` – Photometric reduction.

[inits]
--------

General pipeline settings.

.. list-table::
   :header-rows: 1
   :widths: 20 60 20

   * - Parameter
     - Description
     - Allowed values
   * - ``INSTRUMENT``
     - Name of the instrument.
     - ``TIRSPEC``, ``TANSPEC``
   * - ``TODO``
     - Type of reduction to perform.
     - ``P`` (Photometry), ``S`` (Spectroscopy)
   * - ``DATA``
     - Name of the directory containing the raw observations.
     - Any valid directory
   * - ``EDITOR``
     - Text editor used during manual operations.
     - e.g. ``gedit``, ``nano``
   * - ``TIMESERIES``
     - Enable time-series extraction.
     - ``Y`` or ``N``
   * - ``MODE``
     - Pipeline execution mode.
     - ``AUTO`` or ``MANUAL``

[outputs]
----------

Output directory and filenames.

.. list-table::
   :header-rows: 1

   * - Parameter
     - Description
   * - ``OP_DIR``
     - Output directory where reduced products are stored.
   * - ``EXCLUDE_REG``
     - Comma-separated wildcard patterns of files to ignore.
   * - ``CATALOGUE_NAME``
     - Name of the generated observation catalogue.

[visual]
---------

Image display options.

.. list-table::
   :header-rows: 1

   * - Parameter
     - Description
   * - ``SCIENCE``
     - Display science frames.
   * - ``FLAT``
     - Display flat-field frames.
   * - ``LAMP``
     - Display lamp frames.

All options accept ``Y`` or ``N``.

[logging]
----------

Logging configuration.

.. list-table::
   :header-rows: 1

   * - Parameter
     - Description
   * - ``LEVEL``
     - Logging level. Possible values are ``DEBUG``, ``INFO``,
       ``WARNING`` and ``ERROR``.

[inputs]
---------

Input image processing.

.. list-table::
   :header-rows: 1

   * - Parameter
     - Description
   * - ``FLUXEXT``
     - FITS extension containing the science image.
   * - ``VAREXT``
     - FITS extension containing the variance image.
   * - ``SKY``
     - Perform sky subtraction.
   * - ``REMOVECR``
     - Remove cosmic rays before reduction.
   * - ``BADPIXMASK``
     - Use the default bad-pixel mask (``Y``), disable masking (``N``),
       or specify the path to a custom bad-pixel mask.
   * - ``FRAMECOMBINE``
     - Method used to combine multiple images.

``FRAMECOMBINE`` can be

* ``mean``
* ``median``
* ``biweight``

[dither]
---------

Image alignment parameters.

.. list-table::
   :header-rows: 1

   * - Parameter
     - Description
   * - ``DITHERING``
     - Enable dither reduction.
   * - ``CROP``
     - Region used for determining image shifts.
   * - ``UPSAMPLE``
     - Upsampling factor used for image registration.
   * - ``AUTODITHER``
     - Automatically determine dither shifts.
   * - ``REF_FRAME``
     - Reference frame used for image alignment.

[spectral_extraction]
----------------------

Spectroscopic extraction parameters.

.. list-table::
   :header-rows: 1

   * - Parameter
     - Description
     - Allowed values
   * - ``EXTRACTORCONFIG``
     - Configuration file for SpectrumExtractor.
     - ``Keep blank to use default``
   * - ``SELECT_APERTURE``
     - Aperture selection mode.
     - ``AUTO`` or ``MANUAL``
   * - ``APERTUREWINDOW``
     - Extraction aperture window.
     - e.g. ``-10 10``
   * - ``BKGWINDOWS``
     - Background windows.
     - e.g. ``-20 -15 15 20``
   * - ``SUBTRACT_BKG``
     - Perform background subtraction.
     - ``Y`` or ``N``
   * - ``SCOMBINE``
     - Combine dithered spectra.
     - ``Y`` or ``N``
   * - ``REFITAPERTUREINXD``
     - Refit the aperture in the cross-dispersion direction.
     - e.g. ``p2, p3 etc.``
   * - ``REFITAPERTUREINXD_DWINDOW``
     - Dispersion window used for aperture refitting.
     - e.g. ``(500, 1200)``
   * - ``REFITAPERTUREINXD_BKGMEDIANFILT``
     - Median filter size before aperture refitting.
     - e.g. ``51``
   * - ``DCOEFFMODELFORAPERREFIT``
     - Polynomial model used for aperture refitting.
     - e.g. ``p0``
   * - ``FLUX_CALIB``
     - Apply flux calibration.
     - ``Y`` or ``N``

``REFITAPERTUREINXD`` supports

* ``False`` – Disable refitting.
* ``p0`` – Constant shift.
* ``p1`` – Linear shift.
* ``p2`` – Quadratic shift.
* ``[(coefficients), (xmin, xmax)]`` – Apply a precomputed shift.

[wcs]
------

World Coordinate System parameters.

.. list-table::
   :header-rows: 1

   * - Parameter
     - Description
   * - ``REF_EPOCH``
     - Reference epoch.
   * - ``OBS_EPOCH``
     - Observation epoch.

[photometry]
------------

Photometric reduction parameters.

Source Detection
^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Parameter
     - Description
   * - ``FINDSOURCE``
     - Source detection mode.
   * - ``THRESHOLD``
     - Detection threshold.
   * - ``SOURCELIST``
     - File containing manually selected sources.
   * - ``FWHM``
     - Estimated seeing FWHM in pixels.

``FINDSOURCE`` may be

* ``AUTO``
* ``MANUAL``

Photometry
^^^^^^^^^^

.. list-table::
   :header-rows: 1

   * - Parameter
     - Description
   * - ``METHOD``
     - Photometric extraction method.
   * - ``MODEL``
     - PSF model.
   * - ``PSF_FWHM``
     - Initial PSF FWHM estimate.
   * - ``PSF_ANGLE``
     - PSF rotation angle.
   * - ``APERTURE``
     - Aperture type.
   * - ``ANNULUS``
     - Background annulus type.
   * - ``RADIUS``
     - Aperture radius.
   * - ``BKGWINDOWS``
     - Background annulus dimensions.

Available PSF models

* ``EPSF``
* ``GaussianPSF``
* ``CircularGaussianPSF``

Available aperture types

* ``CircularAperture``
* ``EllipticalAperture``

Available annulus types

* ``CircularAnnulus``
* ``EllipticalAnnulus``

Notes
-----

* Configuration files follow the standard INI format.
* Boolean options use ``Y`` and ``N``.
* Relative paths are interpreted relative to the working directory.
* Values enclosed in parentheses represent Python tuples.
