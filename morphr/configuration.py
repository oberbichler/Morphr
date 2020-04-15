import json
import sys
from morphr.logging import Logger
from collections import OrderedDict
from colorama import init, Fore, Style
from typing import List, Optional
from pydantic import BaseModel
import time


init()


class Configuration:
    def __init__(self, entries):
        self.entries = entries
        self._entry_dict = OrderedDict()
        for entry in entries:
            if entry.key is None:
                continue
            if entry.key in self._entry_dict:
                raise RuntimeError(f'Duplicate key "{entry.key}"')
            self._entry_dict[entry.key] = entry

    def __getitem__(self, key):
        entry = self._entry_dict.get(key, None)
        if entry is None:
            raise KeyError(f'No entry with key "{key}"')
        return entry

    @staticmethod
    def load(path):
        entries = []

        with open(path, 'r') as f:
            for data in json.load(f):
                if not isinstance(data, dict):
                    raise RuntimeError('Entry is not a dictionary')

                type_name = data.get('type', None)

                if type_name is None:
                    raise RuntimeError('Type is missing')

                type_class = getattr(sys.modules['morphr'], type_name)
                entry = type_class(**data)
                entries.append(entry)

        return Configuration(entries)

    def save(self, path):
        data = [entry._to_dict() for entry in self.entries]

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def run(self):
        for job in filter(lambda entry: isinstance(entry, Job), self.entries):
            if job.key.startswith('#'):
                continue
            job.run(self)


class Entry(BaseModel):
    key: Optional[str]

    @classmethod
    def type_name(cls):
        return cls.__name__

    def _to_dict(self):
        data = OrderedDict(type=self.type_name())
        data.update(self.dict(exclude_unset=True, exclude_defaults=True))
        return data


class Job(Entry):
    model_tolerance: float
    info_level: int = 0
    tasks: List[str]
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    def run(self, config):
        log = Logger(info_level=self.info_level)

        log.h1(f'Begin {self.key}...')
        self.start_time = time.perf_counter()

        data = dict(
            cad_model=None,
        )

        DebugData.clear()

        for task_key in self.tasks:
            if task_key.startswith('#'):
                continue
            task = config[task_key]
            log.push(task.info_level)
            task.begin(log)
            task.run(config, self, data, log)
            task.end(log)
            log.pop()

        if not DebugData.is_empty():
            with open('debug_data.json', 'w') as f:
                DebugData.save(f)

        self.end_time = time.perf_counter()
        log.h1(f'Finished {self.key}')
        time_ellapsed = self.end_time - self.start_time
        log.info(f'Done in {time_ellapsed:.2f} sec')


class Task(Entry):
    debug: bool = False
    info_level: Optional[int]
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    @staticmethod
    def select_entries(cad_model, type_name, selector):
        entries = cad_model.of_type(type_name)
        if selector == 'all':
            return entries
        return filter(lambda entry: entry.key in selector, entries)

    def begin(self, log):
        log.h2(Fore.YELLOW + f'{self.key}...' + Style.RESET_ALL)
        self.start_time = time.perf_counter()

    def end(self, log):
        self.end_time = time.perf_counter()
        time_ellapsed = self.end_time - self.start_time
        log.info(f'Done in {time_ellapsed:.2f} sec')


class DebugData:
    _data = list()

    @staticmethod
    def clear():
        DebugData._data.clear()

    @staticmethod
    def is_empty():
        return len(DebugData._data) == 0

    @staticmethod
    def add(**kwargs):
        DebugData._data.append(kwargs)

    @staticmethod
    def tojson():
        return json.dumps(DebugData._data, indent=2)

    @staticmethod
    def save(file):
        return json.dump(DebugData._data, file)
