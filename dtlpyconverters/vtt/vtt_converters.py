from pathlib import Path
import dtlpy as dl
import datetime
import logging
import webvtt
import time
import json
import os
import asyncio
import nest_asyncio

from ..base import BaseExportConverter, BaseImportConverter

logger = logging.getLogger(name='dtlpy')


class VttToDataloop(BaseImportConverter):
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

        nest_asyncio.apply()

        # global vars
        super(VttToDataloop, self).__init__(
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
    def _get_event_loop():
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError as e:
            if "no current event loop" in str(e):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            else:
                raise e
        return loop

    async def convert_dataset(self,
                              speaker_in_caption=True,
                              speaker_seperator=':',
                              speaker_to_annotation=True):
        self.speaker_in_caption = speaker_in_caption
        self.speaker_seperator = speaker_seperator
        self.speaker_to_annotation = speaker_to_annotation
        # inputs
        # read labels and handle recipes
        # with open(self.label_txt_filepath, 'r') as f:
        #     self.id_to_label_map = {i_label: label.strip() for i_label, label in enumerate(f.readlines())}
        # if self.add_labels_to_recipe:
        #     self.dataset.add_labels(label_list=list(self.id_to_label_map.values()))

        # read annotations files and run on items
        files = list(Path(self.input_annotations_path).rglob('*.vtt'))
        files.extend(list(Path(self.input_annotations_path).rglob('*.rst')))

        tasks = [self.on_item(annotation_filepath=str(txt_file)) for txt_file in files]

        # Running the tasks concurrently
        await asyncio.gather(*tasks)

    async def on_item(self, **context):
        """

        """

        annotation_filepath = context.get('annotation_filepath')

        # find images with the same name (ignore image ext)
        relpath = os.path.relpath(annotation_filepath, self.input_annotations_path)
        filename, ext = os.path.splitext(relpath)
        audio_filepaths = list(Path(os.path.join(self.input_items_path)).rglob(f'{filename}.*'))
        if len(audio_filepaths) != 1:
            assert AssertionError

        # image filepath found
        audio_filename = str(audio_filepaths[0])
        remote_rel_path = os.path.relpath(audio_filename, self.input_items_path)
        dirname = os.path.dirname(remote_rel_path)
        if self.upload_items:

            uploader = dl.repositories.uploader.Uploader(items_repository=self.dataset.items)
            item = await uploader._Uploader__single_async_upload(filepath=audio_filename,
                                                                 remote_path=f'/{dirname}',
                                                                 uploaded_filename=f'{os.path.basename(audio_filename)}',
                                                                 last_try=True,
                                                                 mode='skip',
                                                                 item_metadata=dict(),
                                                                 callback=None,
                                                                 item_description=None
                                                                 )
            item = item[0]
        else:
            try:
                item = self.dataset.items.get(f'/{remote_rel_path}')
            except dl.exceptions.NotFound:
                raise ValueError(f'Cannot find corresponding item for annotation: {audio_filename}')
        file, ext = os.path.splitext(annotation_filepath)

        if ext == ".vtt":
            data = await asyncio.to_thread(self.async_read_vtt, annotation_filepath)
        elif ext == ".srt":
            data = await asyncio.to_thread(self.async_from_srt, annotation_filepath)
        else:
            raise ValueError('missing VTT file')

        ignore_speakers = False
        speakers_list = list()
        latest_speaker_index = 0
        speaker_name = None
        speaker_transcript = None

        annotation_collection = item.annotations.builder()
        for caption in data:
            annotation_collection.annotations.append(await self.on_annotation(item=item,
                                                                              caption=caption,
                                                                              ))
        await item.annotations._async_upload_annotations(annotation_collection)
        # print("Updating speaker names for each label")
        # audio_item.metadata["system"]["audioSpeakers"] = {}
        # index = 1
        # for speaker in speakers_list:
        #     audio_item.metadata["system"]["audioSpeakers"]["Speaker {}".format(index)] = speaker
        #     index += 1
        # audio_item.update(system_metadata=True)

    async def on_annotation(self, **context):
        """
        Convert from COCO format to DATALOOP format. Use this as conversion_func param for functions that ask for this param.

        **Prerequisites**: You must be an *owner* or *developer* to use this method.
        :param context: additional params
        :return: converted Annotation entity
        :rtype: dtlpy.entities.annotation.Annotation
        """
        caption = context.get('caption')
        item = context.get('item')

        # convert txt line to yolo params as floats

        if self.speaker_in_caption is True:
            if self.speaker_seperator in caption.raw_text:
                speaker, caption_text = caption.raw_text.split(self.speaker_seperator, 1)
            else:
                logger.warning(f"'speaker_in_caption' flag is on but no seperator found in caption: {caption}")
                speaker, caption_text = None, caption.text

        else:
            speaker, caption_text = None, caption.text

        if speaker is None:
            label = "No Speaker"
        else:
            label = speaker
        if self.speaker_to_annotation is True:
            annotation_text = f"{speaker}: {caption_text}"
        else:
            annotation_text = f"{caption_text}"

        annotation_definition = dl.Subtitle(text=annotation_text, label=label)
        return dl.Annotation.new(annotation_definition=annotation_definition,
                                 item=item,
                                 start_time=caption.start_in_seconds,
                                 end_time=caption.end_in_seconds,
                                 object_id=caption.identifier
                                 )

    @staticmethod
    def async_read_vtt(filepath):
        return webvtt.read(filepath)

    @staticmethod
    def async_from_srt(filepath):
        return webvtt.from_srt(filepath)


class DataloopToVtt(BaseExportConverter):
    """
    Annotation Converter
    """

    async def on_dataset(self, **context) -> dict:
        """

        :param: local_path: directory to save annotations to
        :param context:
        :return:
        """
        from_path = self.dataset.download_annotations(local_path=self.input_annotations_path)
        json_path = Path(from_path).joinpath('json')
        files = list(json_path.rglob('*.json'))
        self.label_to_id_map = self.dataset.instance_map
        os.makedirs(self.output_annotations_path, exist_ok=True)
        sorted_labels = [k for k, v in sorted(self.label_to_id_map.items(), key=lambda item: item[1])]
        with open(os.path.join(self.output_annotations_path, 'labels.txt'), 'w') as f:
            f.write('\n'.join(sorted_labels))

        tic = time.time()
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
                                               annotations=annotations,
                                               to_path=os.path.join(self.output_annotations_path, 'annotations'))
                )
            )
        logger.info('Done converting {} items in {:.2f}[s]'.format(len(files), time.time() - tic))
        return context

    async def on_item(self, **context) -> dict:
        item = context.get('item')
        dataset = context.get('dataset')
        annotations = context.get('annotations')
        to_path = context.get('to_path')
        outputs = dict()
        captions = list()

        for i_annotation, annotation in enumerate(annotations.annotations):
            if annotation.type == dl.AnnotationType.SUBTITLE:
                outs = {"annotation": annotation}
                outs = await self.on_annotation_end(
                    **await self.on_subtitle(
                        **await self.on_annotation_start(**outs)))
                captions.append(outs.get('caption'))
                outputs[annotation.id] = outs
        vtt = webvtt.WebVTT()
        captions = sorted(captions, key=lambda c: c.start)
        vtt.captions.extend(captions)
        name, ext = os.path.splitext(item.name)
        output_filename = os.path.join(to_path, item.dir[1:], name + '.vtt')
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)
        vtt.save(output_filename)
        return context

    async def on_subtitle(self, **context) -> dict:
        """
        Convert from DATALOOP format to YOLO format. Use this as conversion_func param for functions that ask for this param.
        **Prerequisites**: You must be an *owner* or *developer* to use this method.
        :param context:
                See below

        :Keyword Arguments:
            * *annotation* (``dl.Annotations``) -- the box annotations to convert
            * *item* (``dl.Item``) -- Item of the annotation
            * *width* (``int``) -- image width
            * *height* (``int``) -- image height
            * *exif* (``dict``) -- exif information (Orientation)

        :return: converted Annotation
        :rtype: tuple
        """
        annotation = context.get('annotation')

        s = str(datetime.timedelta(seconds=annotation.start_time))
        if len(s.split('.')) == 1:
            s += '.000'
        e = str(datetime.timedelta(seconds=annotation.end_time))
        if len(e.split('.')) == 1:
            e += '.000'
        caption = webvtt.Caption('{}'.format(s),
                                 '{}'.format(e),
                                 '{}'.format(annotation.coordinates['text'])
                                 )
        context['caption'] = caption
        return context
