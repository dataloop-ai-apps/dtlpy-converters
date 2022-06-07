import tempfile
import os
import zipfile
import datetime
import dtlpy as dl
import shutil
import logging
from __version__ import version

logger = logging.getLogger(name=__name__)


class ServiceRunner(dl.BaseServiceRunner):

    def coco(self, dataset, query=None, progress=None):
        return self._convert(dataset=dataset, query=query, to_format=dl.AnnotationFormat.COCO, progress=progress)

    def voc(self, dataset, query=None, progress=None):
        return self._convert(dataset=dataset, query=query, to_format=dl.AnnotationFormat.VOC, progress=progress)

    def yolo(self, dataset, query=None, progress=None):
        return self._convert(dataset=dataset, query=query, to_format=dl.AnnotationFormat.YOLO, progress=progress)

    def coco_no_query(self, dataset, progress=None):
        return self._convert(dataset=dataset, query=None, to_format=dl.AnnotationFormat.COCO, progress=progress)

    def voc_no_query(self, dataset, progress=None):
        return self._convert(dataset=dataset, query=None, to_format=dl.AnnotationFormat.VOC, progress=progress)

    def yolo_no_query(self, dataset, progress=None):
        return self._convert(dataset=dataset, query=None, to_format=dl.AnnotationFormat.YOLO, progress=progress)

    def _convert(self, dataset, query, to_format, progress=None):
        local_path = tempfile.mkdtemp()
        log_filepath = None
        try:
            converted_folder = os.path.join(local_path, to_format)
            os.makedirs(converted_folder, exist_ok=True)
            filters = dl.Filters(resource=dl.FiltersResource.ITEM, custom_filter=query)
            converter = dl.Converter()
            converter.attach_agent_progress(progress=progress, progress_update_frequency=5)
            converter.return_error_filepath = True
            results = converter.convert_dataset(
                dataset=dataset,
                to_format=to_format,
                local_path=converted_folder,
                filters=filters
            )

            if to_format == dl.AnnotationFormat.COCO:
                _, has_errors, log_filepath = results
            else:
                has_errors, log_filepath = results

            if has_errors and log_filepath:
                shutil.copyfile(log_filepath, os.path.join(converted_folder, os.path.basename(log_filepath)))

            try:
                shutil.rmtree(os.path.join(local_path, to_format, 'json'))
            except Exception:
                logger.exception('Failed to delete original annotation folder')

            zip_filename = os.path.join(local_path, '{}_{}_{}.zip'.format(dataset.id, to_format,
                                                                          int(datetime.datetime.now().timestamp())))
            self._zip_directory(zip_filename=zip_filename, directory=converted_folder)

            zip_item = dataset.items.upload(local_path=zip_filename,
                                            remote_path='/.dataloop/converter',
                                            overwrite=True)

            return zip_item.id
        finally:
            shutil.rmtree(local_path)
            if log_filepath is not None:
                os.remove(log_filepath)

    @staticmethod
    def _zip_directory(zip_filename, directory):
        zip_file = zipfile.ZipFile(zip_filename, 'a', zipfile.ZIP_DEFLATED)
        try:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    filepath = os.path.join(root, file)
                    zip_file.write(filepath, arcname=os.path.relpath(filepath, directory))
        finally:
            zip_file.close()


def test():
    dl.setenv('prod')
    ex = dl.executions.get('6151753b9e759075bcadb9db')
    # ex.service.execute(execution_input=ex.input, function_name=ex.function_name)
    self = ServiceRunner()
    self.voc(dataset=dataset)


def get_modules():
    dataset_query_inputs = [
        dl.FunctionIO(name='dataset', type=dl.PackageInputType.DATASET),
        dl.FunctionIO(name='query', type=dl.PackageInputType.JSON)
    ]
    dataset_io = [
        dl.FunctionIO(name='dataset', type=dl.PackageInputType.DATASET)
    ]

    item_io = [
        dl.FunctionIO(name='item', type=dl.PackageInputType.ITEM)
    ]

    module = dl.PackageModule(
        name='yolo-voc-coco',
        entry_point='main_service.py',
        functions=[
            dl.PackageFunction(
                name='coco',
                inputs=dataset_query_inputs,
                outputs=dataset_io,
                description='Converts Dataloop dataset to COCO format'
            ),
            dl.PackageFunction(
                name='yolo',
                inputs=dataset_query_inputs,
                outputs=dataset_io,
                description='Converts Dataloop dataset to YOLO format'
            ),
            dl.PackageFunction(
                name='voc',
                inputs=dataset_query_inputs,
                outputs=dataset_io,
                description='Converts Dataloop dataset to VOC format'
            ),
            dl.PackageFunction(
                name='coco_no_query',
                inputs=dataset_io,
                outputs=dataset_io,
                description='Converts Dataloop dataset to COCO format'
            ),
            dl.PackageFunction(
                name='yolo_no_query',
                inputs=dataset_io,
                outputs=dataset_io,
                description='Converts Dataloop dataset to YOLO format'
            ),
            dl.PackageFunction(
                name='voc_no_query',
                outputs=dataset_io,
                inputs=dataset_io,
                description='Converts Dataloop dataset to VOC format'
            )
        ]
    )
    return [module]


def get_slots():
    slots = [
        dl.PackageSlot(
            function_name='coco',
            module_name='yolo-voc-coco',
            display_icon='fas fa-exchange-alt',
            display_scopes=[
                dl.SlotDisplayScope(resource=dl.SlotDisplayScopeResource.DATASET_QUERY, filters={}),
            ],
            display_name="COCO Converter",
            post_action=dl.SlotPostAction(type=dl.SlotPostActionType.DOWNLOAD)),
        dl.PackageSlot(
            function_name='yolo',
            module_name='yolo-voc-coco',
            display_icon='fas fa-exchange-alt',
            display_scopes=[
                dl.SlotDisplayScope(resource=dl.SlotDisplayScopeResource.DATASET_QUERY, filters={}),
            ],
            display_name="YOLO Converter",
            post_action=dl.SlotPostAction(type=dl.SlotPostActionType.DOWNLOAD)
        ),
        dl.PackageSlot(
            function_name='voc',
            module_name='yolo-voc-coco',
            display_icon='fas fa-exchange-alt',
            display_scopes=[
                dl.SlotDisplayScope(resource=dl.SlotDisplayScopeResource.DATASET_QUERY, filters={}),
            ],
            display_name="VOC Converter",
            post_action=dl.SlotPostAction(type=dl.SlotPostActionType.DOWNLOAD)
        ),

        dl.PackageSlot(
            module_name='yolo-voc-coco',
            function_name='coco_no_query',
            display_icon='fas fa-exchange-alt',
            display_scopes=[
                dl.SlotDisplayScope(resource=dl.SlotDisplayScopeResource.DATASET, filters={})
            ],
            display_name="COCO Converter",
            post_action=dl.SlotPostAction(type=dl.SlotPostActionType.DOWNLOAD)
        ),

        dl.PackageSlot(
            module_name='yolo-voc-coco',
            function_name='yolo_no_query',
            display_icon='fas fa-exchange-alt',
            display_scopes=[
                dl.SlotDisplayScope(resource=dl.SlotDisplayScopeResource.DATASET, filters={})
            ],
            display_name="YOLO Converter",
            post_action=dl.SlotPostAction(type=dl.SlotPostActionType.DOWNLOAD)
        ),

        dl.PackageSlot(
            module_name='yolo-voc-coco',
            function_name='voc_no_query',
            display_icon='fas fa-exchange-alt',
            display_scopes=[
                dl.SlotDisplayScope(resource=dl.SlotDisplayScopeResource.DATASET, filters={})
            ],
            display_name="VOC Converter",
            post_action=dl.SlotPostAction(type=dl.SlotPostActionType.DOWNLOAD)

        )
    ]
    return slots


def deploy():
    env = 'rc'
    dl.setenv(env)
    # dl.login()
    package_name = 'global-converter'
    project_name = 'DataloopTasks'

    project = dl.projects.get(project_name=project_name)

    ################
    # push package #
    ################
    package = project.packages.push(
        package_name=package_name,
        modules=get_modules(),
        slots=get_slots(),
        is_global=True,
        src_path=os.getcwd(),
        version=version
    )

    package = project.packages.get(package_name=package_name)
    service = package.services.get(service_name=package_name)

    ##################
    # create service #
    ##################
    service = package.services.deploy(
        service_name=package.name,
        execution_timeout=60 * 60 * 6,
        is_global=True,
        module_name='yolo-voc-coco',
        sdk_version='1.45.8',
        runtime=dl.KubernetesRuntime(
            concurrency=1,
            pod_type=dl.InstanceCatalog.HIGHMEM_M,
            autoscaler=dl.KubernetesRabbitmqAutoscaler(
                min_replicas=1,
                max_replicas=10,
                queue_length=2
            )
        ),
        bot='pipelines@dataloop.ai'
    )

    service.package_revision = package.version
    service = service.update(True)


def test_service():
    execution_inputs = [
        dl.FunctionIO(name='dataset', type=dl.PackageInputType.DATASET, value='5d8cee9eaa2e03613f207b0b'),
        dl.FunctionIO(name='query', type=dl.PackageInputType.JSON, value={})
    ]
    execution = service.execute(project_id='699d5aea-e8d3-4f7e-a3c9-358a2c60ffc5',
                                execution_input=execution_inputs,
                                function_name='coco')
