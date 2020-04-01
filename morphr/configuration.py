import anurbs as an
import eqlib as eq
import json
import meshio
import numpy as np
import sys
from collections import OrderedDict
from colorama import init, Fore, Style
from typing import List, Optional
from pydantic import BaseModel


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

    def run(self, config):
        print(Fore.GREEN + Style.BRIGHT + f'Begin {self.key}...' + Style.RESET_ALL)
        data = dict(cad_model=None)
        for task_key in self.tasks:
            if task_key.startswith('#'):
                continue
            task = config[task_key]
            task.begin()
            task.run(config, self, data)
        print(Fore.GREEN + Style.BRIGHT + f'Finished {self.key}' + Style.RESET_ALL)


class Task(Entry):
    info_level: Optional[int]

    def log(self, job, message):
        info_level = self.info_level or job.info_level
        if info_level > 0:
            print(message)

    @staticmethod
    def select_entries(cad_model, type_name, selector):
        entries = cad_model.of_type(type_name)
        if selector == 'all':
            return entries
        return filter(lambda entry: entry.key in selector, entries)

    def begin(self):
        print(Fore.YELLOW + f'{self.key}...' + Style.RESET_ALL)
