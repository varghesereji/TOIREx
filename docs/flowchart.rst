Module Flowchart
================

This flowchart shows the main workflow of the package:

.. mermaid::

   flowchart TD

      Start([Start]) --> t0["Task 0<br>Create catalog"]

      t0 --> t1["Task 1<br>Select sci/flat/cal frames"]
      
      t1 --> t2["Task 2<br>Inspect and combine sci<br>frames"]
      
      t2 --> t3["Task 3<br>Inspect and combine<br>flat/cal frames"]

      t3 --> t4["Task 4<br>Flat correction,<br>CR reduction"]

      t4 --> t5["Task 5<br>Dither combine"]

      t5 --> t6["Task 6<br>Spectral extraction<br>Wavelength calibration<br>Flux calibration"]

      t6 --> End([End])
