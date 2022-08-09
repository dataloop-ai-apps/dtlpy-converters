from pathlib import Path
import dtlpy as dl
import numpy as np
import traceback
import logging
import tqdm
import json
import os

from ..base import BaseConverter

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


class DataloopToCoco(BaseConverter):
    def __init__(self, concurrency=6, return_error_filepath=False):
        """
        Dataloop to COCO converter instance
        :param concurrency:
        :param return_error_filepath:


        """
        super(DataloopToCoco, self).__init__(concurrency=concurrency, return_error_filepath=return_error_filepath)
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

    async def convert_dataset(self, dataset, to_path, download_annotations=True, download_images=False, use_rle=True):
        """
        Convert Dataloop Dataset annotation to COCO format

        :param dataset: dl.Dataset entity to convert
        :param to_path: where to save the converted annotation
        :param download_images: download the images with the converted annotations
        :param download_annotations: download annotations from Dataloop or use local
        :param use_rle: convert both segmentation and polygons to RLE encoding.
            if None - default for segmentation is RLE default for polygon is coordinates list
        :return:
        """
        self.dataset = dataset
        self.to_path = to_path
        self.download_images = download_images
        self.download_annotations = download_annotations
        self.use_rle = use_rle

        kwargs = dict()
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
            self.dataset.download_annotations(local_path=self.to_path)
            json_path = Path(self.to_path).joinpath('json')
        else:
            json_path = Path(self.to_path)
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
        """

        :param to_path:
        """
        final_json = {'annotations': list(self.annotations.values()),
                      'categories': list(self.categories.values()),
                      'images': list(self.images.values())}

        os.makedirs(self.to_path, exist_ok=True)
        with open(os.path.join(self.to_path, "coco.json"), 'w') as f:
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
                for point_annotation in point_annotations:
                    if point_annotation.label == pose_point:
                        ordered_points.append(point_annotation)
                        break
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
                    if 'Visible' in list_attributes:
                        keypoints.append(2)
                    elif 'not-visible' in list_attributes or 'not_visible' in list_attributes:
                        keypoints.append(1)
                    else:
                        keypoints.append(0)
                else:
                    keypoints.append(0)
            x_points = keypoints[0::3]
            y_points = keypoints[1::3]
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


class CocoToDataloop:

    def __init__(self):
        ...

    async def convert_dataset(self,
                              dataset,
                              annotation_filepath,
                              upload_images=True,
                              images_path=None,
                              box_only=False,
                              to_polygon=False):

        self.upload_images = upload_images
        self.images_path = images_path
        self.box_only = box_only
        self.to_polygon = to_polygon
        self.dataset = dataset
        self.coco_dataset = pycocotools.coco.COCO(annotation_file=annotation_filepath)
        for coco_image_id, coco_image in self.coco_dataset.imgs.items():
            await self.on_item(coco_image=coco_image)

    async def on_item(self, **kwargs):
        coco_image = kwargs.get('coco_image')

        filename = coco_image['file_name']
        coco_image_id = coco_image['id']
        coco_annotations = self.coco_dataset.imgToAnns[coco_image_id]
        if self.upload_images:
            item = self.dataset.items.upload(os.path.join(self.images_path, filename))
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
