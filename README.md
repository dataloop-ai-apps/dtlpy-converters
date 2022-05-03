# Converters

This is a Dataloop App for three global converters:

1. COCO
1. YOLO
1. VOC

## How This Works

The base class has the following methods for dataset, item and annotation:

* on_dataset_start
* on_dataset
* on_dataset_end
* on_item_start
* on_item
* on_item_end
* on_annotation

For each step, the "start" will perform a pre-process before the actual "on" run. Each "on_<entity>" will thread the
children of the next one to create an optimal runtime. For annotations, there a function "on_<annotation-type>" (e.g "
on_box") to separate the conversion of each type.

## To Dataloop

## From Dataloop

## Run Tests

## Create App (Service)

## Contribute 

