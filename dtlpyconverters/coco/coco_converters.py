import asyncio
from pathlib import Path
import dtlpy as dl
import numpy as np
import traceback
import tqdm
import json
import os
from typing import List

from ..base import BaseExportConverter, BaseImportConverter, logger, get_event_loop

try:
    import pycocotools
except ModuleNotFoundError:
    logger.warning('To use this functionality please install pycocotools: "pip install pycocotools"')
import pycocotools.mask
import pycocotools.coco


class COCOUtils:

    @staticmethod
    def binary_mask_to_rle_encode(binary_mask):
        fortran_ground_truth_binary_mask = np.asfortranarray(binary_mask.astype(np.uint8))
        encoded_ground_truth = pycocotools.mask.encode(fortran_ground_truth_binary_mask)
        encoded_ground_truth['counts'] = encoded_ground_truth['counts'].decode()
        return encoded_ground_truth

    @staticmethod
    def polygon_to_coco_segmentation(geo):
        segmentation = [float(n) for n in geo.flatten()]
        return [segmentation]

    @staticmethod
    def rle_to_binary_mask(rle):
        rows, cols = rle['size']
        rle_numbers = rle['counts']
        if isinstance(rle_numbers, list):
            if len(rle_numbers) % 2 != 0:
                rle_numbers.append(0)

            rle_pairs = np.array(rle_numbers).reshape(-1, 2)
            img = np.zeros(rows * cols, dtype=np.uint8)
            index = 0
            for i, length in rle_pairs:
                index += i
                img[index:index + length] = 1
                index += length
            img = img.reshape(cols, rows)
            return img.T
        else:
            img = pycocotools.mask.decode(rle)
            return img


class DataloopToCoco(BaseExportConverter):
    def __init__(self,
                 dataset: dl.Dataset,
                 output_annotations_path,
                 output_items_path=None,
                 input_annotations_path=None,
                 filters: dl.Filters = None,
                 download_annotations=True,
                 download_items=False,
                 concurrency=6,
                 return_error_filepath=False,
                 label_to_id_mapping=None):
        """
        Convert Dataloop Dataset annotation to COCO format.

        :param dataset: dl.Dataset entity to convert
        :param output_annotations_path: where to save the converted annotations json
        :param output_items_path: where to save the downloaded items
        :param input_annotations_path: where to save the downloaded dataloop annotations files. Default is output_annotations_path
        :param filters: dl.Filters object to filter the items from dataset
        :param download_items: download the images with the converted annotations
        :param download_annotations: download annotations from Dataloop or use local
        :param label_to_id_mapping: dictionary to map labels to ids
        :return:
        """
        # global vars
        super(DataloopToCoco, self).__init__(
            dataset=dataset,
            output_annotations_path=output_annotations_path,
            output_items_path=output_items_path,
            input_annotations_path=input_annotations_path,
            filters=filters,
            download_annotations=download_annotations,
            download_items=download_items,
            concurrency=concurrency,
            return_error_filepath=return_error_filepath,
            label_to_id_mapping=label_to_id_mapping,
        )
        # COCO related
        self.images = dict()
        self.categories = dict()
        self.annotations = dict()

    @staticmethod
    def gen_coco_categories(instance_map, recipe):
        """
        Generate COCO category map from the dataset's ontology
        """
        categories = list()
        last_id = 0
        for label, label_id in instance_map.items():
            label_name, sup = label.split('.')[-1], '.'.join(label.split('.')[0:-1])
            category = {'id': label_id, 'name': label_name}
            last_id = max(last_id, label_id)
            if sup:
                category['supercategory'] = sup
            categories.append(category)

        # add keypoint category
        collection_templates = list()
        if 'system' in recipe.metadata and 'collectionTemplates' in recipe.metadata['system']:
            collection_templates = recipe.metadata['system']['collectionTemplates']

        for template in collection_templates:
            last_id += 1
            order_dict = {key: i for i, key in enumerate(template['order'])}
            skeleton = list()
            for pair in template['arcs']:
                skeleton.append([order_dict[pair[0]], order_dict[pair[1]]])
            category = {'id': last_id,
                        'name': template['name'],
                        'templateId': template['id'],
                        'keypoints': template['order'],
                        'skeleton': skeleton}
            instance_map[template['name']] = last_id
            categories.append(category)

        return categories

    async def convert_dataset(self, **kwargs):
        """
        Convert Dataloop Dataset annotations to COCO format.
        :param use_rle: convert both segmentation and polygons to RLE encoding.
            if None - default for segmentation is RLE default for polygon is coordinates list
        :return:
        """
        self.use_rle = kwargs.get("use_rle", True)
        return await self.on_dataset(**kwargs)

    async def on_dataset(self, **kwargs):
        """
        Callback to run the conversion on a dataset.
        Will be called after on_dataset_start and before on_dataset_end.
        """
        kwargs = await self.on_dataset_start(**kwargs)
        if self.download_annotations:
            self.dataset.download_annotations(local_path=self.input_annotations_path,
                                              filters=self.filters)
            json_path = Path(self.input_annotations_path).joinpath('json')
        else:
            json_path = Path(self.input_annotations_path)
        if self.download_items:
            self.dataset.items.download(local_path=self.output_items_path)

        files = list(json_path.rglob('*.json'))
        self.categories = {cat['name']: cat for cat in self.gen_coco_categories(self.label_to_id_mapping,
                                                                                self.dataset.recipes.list()[0])}
        self.pbar = tqdm.tqdm(total=len(files))
        futures = list()
        for annotation_json_filepath in files:
            with open(annotation_json_filepath, 'r') as f:
                data = json.load(f)
            json_annotations = data.pop('annotations')
            item = dl.Item.from_json(_json=data,
                                     client_api=dl.client_api,
                                     dataset=self.dataset)
            annotations = dl.AnnotationCollection.from_json(_json=json_annotations, item=item)
            futures.append(asyncio.create_task(self.on_item(item=item,
                                                            dataset=self.dataset,
                                                            annotations=annotations)))
        await asyncio.gather(*futures)
        kwargs = await self.on_dataset_end(**kwargs)
        return kwargs

    async def on_dataset_end(self, **kwargs):
        final_json = {'annotations': list(self.annotations.values()),
                      'categories': list(self.categories.values()),
                      'images': list(self.images.values())}

        os.makedirs(self.output_annotations_path, exist_ok=True)
        with open(os.path.join(self.output_annotations_path, "coco.json"), 'w') as f:
            json.dump(final_json, f, indent=2)
        return kwargs

    async def on_item(self, **kwargs):
        """

        :param item:
        :param annotations:
        """
        kwargs = await self.on_item_start(**kwargs)

        item = kwargs.get('item')
        annotations = kwargs.get('annotations')
        logger.debug(f'Started: {item.id}')

        self.images[item.id] = {'file_name': item.filename[1:],
                                'id': item.id,
                                'width': item.width,
                                'height': item.height
                                }

        for i_annotation, annotation in enumerate(annotations.annotations):
            context = dict(annotation=annotation,
                           annotations=annotations,
                           item=item)
            context = await self.on_annotation_start(**context)
            if annotation.type == dl.AnnotationType.BOX:
                context = await self.on_box(**context)
            elif annotation.type == dl.AnnotationType.POSE:
                context = await self.on_pose(**context)
            elif annotation.type == dl.AnnotationType.POLYGON:
                context = await self.on_polygon(**context)
            elif annotation.type == dl.AnnotationType.SEGMENTATION:
                context = await self.on_segmentation(**context)
            context = await self.on_annotation_end(**context)

        kwargs = await self.on_item_end(**kwargs)
        logger.debug(f'Done: {item.id}')
        return kwargs

    async def on_item_end(self, **kwargs):
        """

        """
        self.pbar.update()
        return kwargs

    ##################
    # on annotations #
    ##################
    async def on_point(self, **kwargs):
        # handled in the pose. single point is not supported
        ...

    async def on_pose(self, **kwargs):
        """
        :param kwargs: See below

        * item (``dl.Item``) -- the current item
        * annotation (``list``) -- the current annotation
        * annotations (``dl.AnnotationCollection``) -- the entire annotation collection of the item

        """
        try:
            annotation = kwargs.get('annotation')
            annotations = kwargs.get('annotations')
            item = kwargs.get('item')
            iscrowd = 0
            segmentation = [[]]
            if annotation.type not in ['binary', 'box', 'segment', 'pose']:
                return

            #########
            # Pose
            pose_category = None
            for category in self.categories:
                if annotation.coordinates.get('templateId', "") == self.categories[category].get('templateId', None):
                    pose_category = category
                    continue
            if pose_category is None:
                err = 'Error converting annotation: \n' \
                      'Item: {}, annotation: {} - ' \
                      'Pose annotation without known template\n{}'.format(item.id,
                                                                          annotation.id,
                                                                          traceback.format_exc())
                raise ValueError(err)
            ordered_points = list()
            point_annotations = [ann for ann in annotations if ann.parent_id == annotation.id]
            for pose_point in self.categories[pose_category]['keypoints']:
                ordered_point_found = False
                for point_annotation in point_annotations:
                    if point_annotation.label == pose_point:
                        ordered_points.append(point_annotation)
                        ordered_point_found = True
                        break
                if not ordered_point_found:
                    # if points doenst exists - create dummy one for order. add not-visible attribute
                    missing_point = dl.Point(label=pose_point,
                                             x=0.0,
                                             y=0.0)
                    if isinstance(missing_point.attributes, list):
                        missing_point.attributes.append('not-visible')
                    elif isinstance(missing_point.attributes, dict):
                        missing_point.attributes['visibility'] = 'not-visible'
                    else:
                        raise ValueError('Unknown point.attributes type: {}'.format(type(missing_point.attributes)))
                    ordered_points.append(missing_point)
            keypoints = list()
            for point in ordered_points:
                keypoints.append(point.x)
                keypoints.append(point.y)
                # v=0 not labeled , v=1: labeled but not visible, and v=2: labeled and visible
                if isinstance(point.attributes, list):
                    if 'visible' in point.attributes and \
                            ("not-visible" in point.attributes or 'not_visible' in point.attributes):
                        keypoints.append(0)
                    elif 'not-visible' in point.attributes or 'not_visible' in point.attributes:
                        keypoints.append(1)
                    elif 'visible' in point.attributes:
                        keypoints.append(2)
                    else:
                        keypoints.append(0)
                elif isinstance(point.attributes, dict):
                    list_attributes = list(point.attributes.values())
                    if 'visible' in list_attributes:
                        keypoints.append(2)
                    elif 'not-visible' in list_attributes or 'not_visible' in list_attributes:
                        keypoints.append(1)
                    else:
                        keypoints.append(0)
                else:
                    keypoints.append(0)
            # get bounding box from existing points only
            x_points = [pt.x for pt in point_annotations]
            y_points = [pt.y for pt in point_annotations]
            x0, x1, y0, y1 = np.min(x_points), np.max(x_points), np.min(y_points), np.max(y_points)
            x = float(x0)
            y = float(y0)
            w = float(x1 - x)
            h = float(y1 - y)
            area = (x1 - x0) * (y1 - y0)
            ann = dict()
            ann['bbox'] = [float(x), float(y), float(w), float(h)]
            ann["segmentation"] = segmentation
            ann["area"] = area
            ann["iscrowd"] = iscrowd
            if keypoints is not None:
                ann["keypoints"] = keypoints
            ann['category_id'] = self.categories[annotation.label]['id']
            ann['image_id'] = self.images[item.id]['id']
            ann['id'] = annotation.id
            self.annotations[annotation.id] = ann
            return kwargs

        except Exception:
            print(traceback.format_exc())

    async def on_box(self, **kwargs):
        """
        :param item:
        :param annotation:
        """
        try:
            annotation = kwargs.get('annotation')
            item = kwargs.get('item')

            height = item.height if item is not None else None
            width = item.width if item is not None else None
            iscrowd = 0
            segmentation = [[]]
            if annotation.type in ['binary', 'segment']:
                if height is None or width is None:
                    raise Exception(
                        'Item must have height and width to convert {!r} annotation to coco'.format(annotation.type))

            # build annotation
            keypoints = None
            if annotation.type not in ['binary', 'box', 'segment', 'pose']:
                return
            x = float(annotation.left)
            y = float(annotation.top)
            w = float(annotation.right - x)
            h = float(annotation.bottom - y)
            area = h * w
            ann = dict()
            ann['bbox'] = [float(x), float(y), float(w), float(h)]
            ann["segmentation"] = segmentation
            ann["area"] = area
            ann["iscrowd"] = iscrowd
            if keypoints is not None:
                ann["keypoints"] = keypoints
            category = annotation.label.split('.')[-1]
            try:
                ann['category_id'] = self.categories[category]['id']
            except KeyError as e:
                raise KeyError(f"Category {category} not found in dataset for label {annotation.label}") from e
            ann['image_id'] = self.images[item.id]['id']
            ann['id'] = annotation.id
            self.annotations[annotation.id] = ann
            return kwargs

        except Exception:
            print(traceback.format_exc())
            return kwargs

    async def on_segmentation(self, **kwargs):
        """
        :param item:
        :param annotation:
        """
        try:
            annotation = kwargs.get('annotation')
            item = kwargs.get('item')

            height = item.height if item is not None else None
            width = item.width if item is not None else None
            if annotation.type in ['binary', 'segment']:
                if height is None or width is None:
                    raise Exception(
                        'Item must have height and width to convert {!r} annotation to coco'.format(annotation.type))

            # build annotation
            keypoints = None
            x = float(annotation.left)
            y = float(annotation.top)
            w = float(annotation.right - x)
            h = float(annotation.bottom - y)

            area = int(annotation.geo.sum())

            if self.use_rle is False:
                pol_annotation = dl.Polygon.from_segmentation(mask=annotation.geo,
                                                              max_instances=1,
                                                              label=None)
                segmentation = COCOUtils.polygon_to_coco_segmentation(geo=pol_annotation.geo)
                iscrowd = 0  # https://github.com/cocodataset/cocoapi/issues/135
            else:
                # None or True
                segmentation = COCOUtils.binary_mask_to_rle_encode(binary_mask=annotation.geo)
                iscrowd = 1  # https://github.com/cocodataset/cocoapi/issues/135

            ann = dict()
            ann['bbox'] = [float(x), float(y), float(w), float(h)]
            ann["segmentation"] = segmentation
            ann["area"] = area
            ann["iscrowd"] = iscrowd
            if keypoints is not None:
                ann["keypoints"] = keypoints
            category = annotation.label.split('.')[-1]
            try:
                ann['category_id'] = self.categories[category]['id']
            except KeyError:
                raise KeyError(f"Category {category} not found in dataset for label {annotation.label}")
            ann['image_id'] = self.images[item.id]['id']
            ann['id'] = annotation.id
            self.annotations[annotation.id] = ann
            return kwargs

        except Exception:
            print(traceback.format_exc())

    async def on_polygon(self, **kwargs):
        """
        :param item:
        :param annotation:
        """
        try:
            annotation = kwargs.get('annotation')
            item = kwargs.get('item')
            height = item.height if item is not None else None
            width = item.width if item is not None else None
            if height is None or width is None:
                raise Exception(
                    'Item must have height and width to convert {!r} annotation to coco'.format(annotation.type))

            # build annotation
            keypoints = None
            if annotation.type not in ['binary', 'box', 'segment', 'pose']:
                return
            x = float(annotation.left)
            y = float(annotation.top)
            w = float(annotation.right - x)
            h = float(annotation.bottom - y)

            seg_annotation = dl.Segmentation.from_polygon(geo=annotation.geo,
                                                          label=None,
                                                          shape=(height, width))
            area = int(seg_annotation.geo.sum())
            if self.use_rle is True:
                segmentation = COCOUtils.binary_mask_to_rle_encode(binary_mask=seg_annotation.geo)
                iscrowd = 0  # https://github.com/cocodataset/cocoapi/issues/135
            else:
                # None or False
                segmentation = COCOUtils.polygon_to_coco_segmentation(geo=annotation.geo)
                iscrowd = 1  # https://github.com/cocodataset/cocoapi/issues/135

            ann = dict()
            ann['bbox'] = [float(x), float(y), float(w), float(h)]
            ann["segmentation"] = segmentation
            ann["area"] = area
            ann["iscrowd"] = iscrowd
            if keypoints is not None:
                ann["keypoints"] = keypoints
            ann['image_id'] = self.images[item.id]['id']
            ann['id'] = annotation.id
            category = annotation.label.split('.')[-1]
            try:
                ann['category_id'] = self.categories[category]['id']
            except KeyError:
                raise KeyError(f"Category {category} not found in dataset for label {annotation.label}")
            self.annotations[annotation.id] = ann
            return kwargs

        except Exception:
            print(traceback.format_exc())

    async def on_polyline(self, **kwargs):
        raise Exception('Unable to convert annotation of type "polyline" to coco')

    async def on_class(self, **kwargs):
        raise Exception('Unable to convert annotation of type "class" to coco')


class CocoToDataloop(BaseImportConverter):

    def __init__(self,
                 dataset: dl.Dataset,
                 input_annotations_path,
                 output_annotations_path=None,
                 input_items_path=None,
                 upload_items=False,
                 add_labels_to_recipe=True,
                 concurrency=6,
                 return_error_filepath=False,
                 ):
        # global vars
        super(CocoToDataloop, self).__init__(
            dataset=dataset,
            output_annotations_path=output_annotations_path,
            input_annotations_path=input_annotations_path,
            input_items_path=input_items_path,
            upload_items=upload_items,
            add_labels_to_recipe=add_labels_to_recipe,
            concurrency=concurrency,
            return_error_filepath=return_error_filepath,
        )

    @staticmethod
    def create_label_hierarchy(categories):
        """
        Creates a dictionary mapping category IDs to their supercategory hierarchy and name.

        Returns:
            A dictionary mapping category IDs to their supercategory hierarchy and name.
        """
        labels = {}
        for category_id, category in categories.items():
            name = category["name"]
            super_category = category.get("supercategory")
            hierarchy = []
            if name == super_category:
                hierarchy.append(super_category)
            else:
                while super_category is not None:
                    hierarchy.append(super_category)
                    super_category = None
                    for cat_id, cat in categories.items():
                        if cat["name"] == hierarchy[-1]:
                            super_category = cat.get("supercategory", None)
                            break
            hierarchy.reverse()
            hierarchy.append(name)
            labels[category_id] = ".".join(hierarchy)
        return labels

    def convert(self,
                annotation_options: List[dl.AnnotationType] = None,
                coco_json_filename='coco.json',
                to_polygon=False):
        """
        Sync call to 'convert_dataset'.
        :param annotation_options: dataloop annotation type options to export from: SEGMENTATION, POSE and BOX (by default: BOX)
        :param coco_json_filename: coco json filename
        :param to_polygon:
        :return:
        """
        loop = get_event_loop()
        loop.run_until_complete(future=self.convert_dataset(
            annotation_options=annotation_options,
            coco_json_filename=coco_json_filename,
            to_polygon=to_polygon
        ))

    async def convert_dataset(self,
                              annotation_options: List[dl.AnnotationType] = None,
                              coco_json_filename='coco.json',
                              to_polygon=False):
        """
        Converting a dataset from COCO format to Dataloop.
        :param annotation_options: dataloop annotation type options to export from: SEGMENTATION, POSE and BOX (by default: BOX)
        :param coco_json_filename: coco json filename
        :param to_polygon:
        :return:
        """
        self.annotation_options = annotation_options if annotation_options is not None else [dl.AnnotationType.BOX]
        self.to_polygon = to_polygon
        self.coco_dataset = pycocotools.coco.COCO(
            annotation_file=os.path.join(self.input_annotations_path, coco_json_filename))
        self.labels = self.create_label_hierarchy(self.coco_dataset.cats)
        futures = [asyncio.create_task(self.on_item(coco_image=coco_image))
                   for coco_image_id, coco_image in self.coco_dataset.imgs.items()]
        await asyncio.gather(*futures)

        if self.add_labels_to_recipe is True:
            self.dataset.add_labels(label_list=list(self.labels.values()))

    async def on_item(self, **kwargs):
        coco_image = kwargs.get('coco_image')
        filename = coco_image['file_name']
        coco_image_id = coco_image['id']
        logger.debug(f'Started: {coco_image_id}')
        coco_annotations = self.coco_dataset.imgToAnns[coco_image_id]
        if self.upload_items:
            uploader = dl.repositories.uploader.Uploader(items_repository=self.dataset.items)
            item = await uploader._Uploader__single_async_upload(filepath=os.path.join(self.input_items_path, filename),
                                                                 remote_path=f'/{os.path.dirname(filename)}',
                                                                 uploaded_filename=f'{os.path.basename(filename)}',
                                                                 last_try=True,
                                                                 mode='skip',
                                                                 item_metadata=dict(),
                                                                 callback=None,
                                                                 item_description=None
                                                                 )
            item = item[0]
        else:
            item = self.dataset.items.get(f'/{filename}')

        annotation_collection = item.annotations.builder()
        for coco_annotation in coco_annotations:
            new_annotation = await self.on_annotation(item=item,
                                                      coco_annotation=coco_annotation)
            if new_annotation is not None:
                annotation_collection.annotations.append(new_annotation)
        # for async uploading
        await item.annotations._async_upload_annotations(annotation_collection)
        logger.debug(f'Done: {coco_image_id}')

    async def on_annotation(self, **kwargs):
        """
        Convert from COCO format to DATALOOP format. Use this as conversion_func param for functions that ask for this param.

        **Prerequisites**: You must be an *owner* or *developer* to use this method.

        :param kwargs: additional params
        :return: converted Annotation entity
        :rtype: dtlpy.entities.annotation.Annotation
        """
        coco_annotation = kwargs.get('coco_annotation')
        item = kwargs.get('item')
        coco_annotation_id = coco_annotation.get('id', None)
        category_id = coco_annotation.get('category_id', None)
        segmentation = coco_annotation.get('segmentation', None)
        iscrowd = coco_annotation.get('iscrowd', None)
        keypoints = coco_annotation.get('keypoints', None)
        bbox = coco_annotation.get('bbox', None)
        label = self.labels[category_id]

        ann_def = None
        if segmentation is not None and dl.AnnotationType.SEGMENTATION in self.annotation_options:
            # upload semantic as binary or polygon
            if isinstance(segmentation, dict):
                mask = COCOUtils.rle_to_binary_mask(segmentation)
                if self.to_polygon is True:
                    ann_def = dl.Polygon.from_segmentation(label=label,
                                                           mask=mask)
                else:
                    ann_def = dl.Segmentation(label=label,
                                              geo=mask)
            else:
                if len(segmentation) > 1:
                    logger.warning('Multiple polygons per annotation is not supported. coco annotation id: {}'.format(
                        coco_annotation_id))
                if len(segmentation) < 1:
                    segmentation = [[]]
                    logger.warning(
                        'Empty segmentation, using default: [[]]. coco annotation id: {}'.format(coco_annotation_id))
                segmentation = np.reshape(segmentation[0], (-1, 2))
                # if segmentation is empty, the annotation is box
                if len(segmentation) == 0:
                    ann_def = None
                elif self.to_polygon is False:
                    ann_def = dl.Segmentation.from_polygon(label=label,
                                                           geo=segmentation,
                                                           shape=(item.height, item.width))
                else:
                    ann_def = dl.Polygon(label=label,
                                         geo=segmentation)

        if keypoints is not None and dl.AnnotationType.POSE in self.annotation_options:
            # upload keypoints
            ann_def = None
            logger.warning('keypoints is not supported yet')
        if ann_def is None and dl.AnnotationType.BOX in self.annotation_options:
            # upload box only
            left = bbox[0]
            top = bbox[1]
            right = left + bbox[2]
            bottom = top + bbox[3]
            ann_def = dl.Box(top=top,
                             left=left,
                             bottom=bottom,
                             right=right,
                             label=label)
        new_ann = None
        if ann_def is not None:
            new_ann = dl.Annotation.new(annotation_definition=ann_def, item=item)
        return new_ann
