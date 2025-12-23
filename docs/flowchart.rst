Module Flowchart
================

This flowchart shows the main workflow of the package:

.. mermaid::

   %%{init: {'flowchart': {'wrap': true}} }%%
   flowchart TD

      Start([Start]) --> t0["Task 0
Create catalog"]

      t0 --> t1["Task 1
Select sci/flat/cal frames"]
      
      t1 --> t2["Task 2
Inspect and combine sci
frames"]
      
      t2 --> t3["Task 3
Inspect and combine
flat/cal frames"]

      t3 --> t4["Task 4
Flat correction,
CR reduction"]

      t4 --> t5["Task 5
Dither combine"]

      t5 --> t6["Task 6
Spectral extraction
Wavelength calibration
Flux calibration"]

      t6 --> End([End])
