# Generic Multiprocessing Generator

This project implements a **generic** multiprocessing generator for machine learning training processes (also can be used for any other purposes).

It allows integration of domain-specific data-readers, preprocessors, sample-weighters etc.

**TO-DO**
- Add tests to ensure robustness (Current test shows that the number of generated batch size holds as expected. However, need more fine-grained test).
- Add exception handling
- Add signal handling (e.g. SIGTERM, SIGINT etc.)
- Add performance results (pre-experiments shows a linear increase in timing as the number of workers increases. The effect gets higher as processing (reading data, preprocessing etc.) a sample/batch takes longer).
- Add documentation (though the existing comments will be guiding well enough for now)