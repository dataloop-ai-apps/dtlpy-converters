
## Dataloop converters base architecture

The base class has the following methods for dataset, item and annotation:

* on_dataset_start
* on_dataset
* on_dataset_end
* on_item_start
* on_item
* on_item_end
* on_annotation

For each step, the "on_{entity}\_start" will perform a pre-process before the actual "on_{entity}" run. Each "on_
{entity}" will thread the children of the next level to create an optimal runtime. For annotations, there's a function "
on_{annotation-type}" (e.g "on_box") to separate the conversion of each type.

The following diagram demonstrate the pre/post functions and the parallelism of the conversion:
![diagram](parallel_diagram.png)
