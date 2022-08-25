from pathlib import Path
import dtlpy as dl
import numpy as np
import traceback
import logging
import tqdm
import json
import os

from ..base import BaseExportConverter, BaseImportConverter

try:
    import pycocotools
except ModuleNotFoundError:
    raise Exception('To use this functionality please install pycocotools: "pip install pycocotools"')
import pycocotools.mask
import pycocotools.coco

logger = logging.getLogger(__name__)


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
                 input_annotations_path=None,
                 filters: dl.Filters = None,
                 download_annotations=True,
                 download_items=False,
                 concurrency=6,
                 return_error_filepath=False):
        """
        Convert Dataloop Dataset annotation to COCO format.

        :param dataset: dl.Dataset entity to convert
        :param output_annotations_path: where to save the converted annotations json
        :param input_annotations_path: where to save the downloaded dataloop annotations files. Default is output_annotations_path
        :param filters: dl.Filters object to filter the items from dataset
        :param download_items: download the images with the converted annotations
        :param download_annotations: download annotations from Dataloop or use local
        :return:
        """
        # global vars
        super(DataloopToCoco, self).__init__(
            dataset=dataset,
            output_annotations_path=output_annotations_path,
            input_annotations_path=input_annotations_path,
            filters=filters,
            download_annotations=download_annotations,
            download_items=download_items,
            concurrency=concurrency,
            return_error_filepath=return_error_filepath,
        )
        # COCO related
        self.images = dict()
        self.categories = dict()
        self.annotations = dict()

        self.pose_with_attributes = False

        # For Pose:
        # Check if Ontology works with visible/not-visible attributes for unexisting point
        if len(dataset.ontology_ids):
            ontology = dataset.ontologies.get(ontology_id=dataset.ontology_ids[0])
            ontology_attributes = ontology.metadata.get("attributes", dict())
            for attribute in ontology_attributes:
                if isinstance(attribute, dict):
                    for value in attribute['values']:
                        if value == 'visible':
                            self.pose_with_attributes = True
                            break
                    if self.pose_with_attributes:
                        break

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
        Convert Dataloop Dataset annotation to COCO format.
        :param use_rle: convert both segmentation and polygons to RLE encoding.
            if None - default for segmentation is RLE default for polygon is coordinates list
        :return:
        """
        self.use_rle = kwargs.get('use_rle', True)
        return await self.on_dataset_end(
            **await self.on_dataset(
                **await self.on_dataset_start(**kwargs)
            )
        )

    async def on_dataset(self, **kwargs):
        """
        Callback to tun the conversion on a dataset.
        Will be called after on_dataset_start and before on_dataset_end
        """

        if self.download_annotations:
            self.dataset.download_annotations(local_path=self.input_annotations_path,
                                              filters=self.filters)
            json_path = Path(self.input_annotations_path).joinpath('json')
        else:
            json_path = Path(self.input_annotations_path)
        files = list(json_path.rglob('*.json'))
        self.categories = {cat['name']: cat for cat in self.gen_coco_categories(self.dataset.instance_map,
                                                                                self.dataset.recipes.list()[0])}

        self.pbar = tqdm.tqdm(total=len(files))
        for annotation_json_filepath in files:
            with open(annotation_json_filepath, 'r') as f:
                data = json.load(f)
                json_annotations = data.pop('annotations')
                item = dl.Item.from_json(_json=data,
                                         client_api=dl.client_api,
                                         dataset=self.dataset)
                annotations = dl.AnnotationCollection.from_json(_json=json_annotations, item=item)
                _ = await self.on_item_end(
                    **await self.on_item(
                        **await self.on_item_start(item=item,
                                                   dataset=self.dataset,
                                                   annotations=annotations)
                    )
                )

        return kwargs

    async def on_dataset_end(self, **kwargs):
        final_json = {'annotations': list(self.annotations.values()),
                      'categories': list(self.categories.values()),
                      'images': list(self.images.values())}

        # Generate output file only in case that there are images
        if len(final_json['images']):
            os.makedirs(self.output_annotations_path, exist_ok=True)
            with open(os.path.join(self.output_annotations_path, "coco.json"), 'w') as f:
                json.dump(final_json, f, indent=2)

    async def on_item(self, **kwargs):
        """

        :param item:
        :param annotations:
        """
        item = kwargs.get('item')
        annotations = kwargs.get('annotations')

        self.images[item.id] = {'file_name': item.name,
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
                has_point = False
                for point_annotation in point_annotations:
                    if point_annotation.label == pose_point:
                        ordered_points.append(point_annotation)
                        has_point = True
                        break
                if not has_point:
                    missing_point = dl.Point(label=pose_point,x=0.0, y=0.0, description="missing point")
                    ordered_points.append(missing_point)
            keypoints = list()
            existing_keypoints = list()
            for point in ordered_points:
                keypoints.append(point.x)
                keypoints.append(point.y)
                if point.description != "missing point":
                    existing_keypoints.append(point.x)
                    existing_keypoints.append(point.y)
                    existing_keypoints.append(2)
                # v=0 not labeled , v=1: labeled but not visible, and v=2: labeled and visible
                if self.pose_with_attributes:
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
                else:
                    if point.description == "missing point":
                        keypoints.append(1)
                    else:
                        keypoints.append(2)
            x_points = existing_keypoints[0::3]
            y_points = existing_keypoints[1::3]
            try:
                x0, x1, y0, y1 = np.min(x_points), np.max(x_points), np.min(y_points), np.max(y_points)
            except:
                x0 = x1 = y0 = y1 = 0
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
            label = annotation.label
            ann['category_id'] = self.dataset.instance_map[label]
            ann['image_id'] = self.images[item.id]['id']
            ann['id'] = annotation.id
            self.annotations[annotation.id] = ann
            return kwargs

        except Exception:
            print(traceback.format_exc())

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
            ann['category_id'] = self.categories[annotation.label]['id']
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
            ann['category_id'] = self.categories[annotation.label]['id']
            ann['image_id'] = self.images[item.id]['id']
            ann['id'] = annotation.id
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

    async def convert_dataset(self,
                              box_only=False,
                              coco_json_filename='coco.json',
                              to_polygon=False):
        self.box_only = box_only
        self.to_polygon = to_polygon
        self.coco_dataset = pycocotools.coco.COCO(annotation_file=os.path.join(self.input_annotations_path, coco_json_filename))
        for coco_image_id, coco_image in self.coco_dataset.imgs.items():
            await self.on_item(coco_image=coco_image)

    async def on_item(self, **kwargs):
        coco_image = kwargs.get('coco_image')

        filename = coco_image['file_name']
        coco_image_id = coco_image['id']
        coco_annotations = self.coco_dataset.imgToAnns[coco_image_id]
        if self.upload_items:
            item = self.dataset.items.upload(os.path.join(self.input_items_path, filename))
        else:
            item = self.dataset.items.get(f'/{filename}')

        annotation_collection = item.annotations.builder()
        for coco_annotation in coco_annotations:
            annotation_collection.annotations.append(await self.on_annotation(item=item,
                                                                              coco_annotation=coco_annotation))
        item.annotations.upload(annotation_collection)

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
        label = self.coco_dataset.cats[category_id]['name']

        if segmentation is not None and not self.box_only:
            # upload semantic as binary or polygon
            if isinstance(segmentation, dict):
                mask = COCOUtils.rle_to_binary_mask(segmentation)
                if self.to_polygon:
                    ann_def = dl.Polygon.from_segmentation(label=label,
                                                           mask=mask)
                else:
                    ann_def = dl.Segmentation(label=label,
                                              geo=mask)
            else:
                if len(segmentation) > 1:
                    logger.warning('Multiple polygons per annotation is not supported. coco annotation id: {}'.format(
                        coco_annotation_id))
                segmentation = np.reshape(segmentation[0], (-1, 2))
                polygon = segmentation
                if self.to_polygon:
                    ann_def = dl.Segmentation.from_polygon(label=label,
                                                           geo=polygon,
                                                           shape=(item.height, item.width))
                else:
                    ann_def = dl.Polygon(label=label,
                                         geo=polygon)

        elif keypoints is not None and not self.box_only:
            # upload keypoints
            raise ValueError('keypoints is not supported yet')
        else:
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
        return dl.Annotation.new(annotation_definition=ann_def, item=item)
