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

                type_class = getattr(sys.modules[__name__], type_name)
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


class ImportDisplacementField(Task):
    mesh_0: str
    mesh_1: str

    def run(self, config, job, data):
        mesh_0 = meshio.read(self.mesh_0, file_format='obj')
        mesh_1 = meshio.read(self.mesh_1, file_format='obj')

        nb_points = len(mesh_0.points)

        vertices = np.empty((nb_points, 3), float)
        displacements = np.empty((nb_points, 3), float)

        for i, (point_0, point_1) in enumerate(zip(mesh_0.points, mesh_1.points)):
            vertices[i] = point_0
            displacements[i] = np.subtract(point_1, point_0)

        faces = []

        if 'triangle' in mesh_0.cells:
            for a, b, c in mesh_0.cells['triangle']:
                faces.append([a, b, c])

        if 'quad' in mesh_0.cells:
            for a, b, c, d in mesh_0.cells['quad']:
                faces.append([a, b, c])
                faces.append([c, d, a])

        data['vertices'] = vertices
        data['displacements'] = displacements
        data['faces'] = faces


class ExportMdpa(Task):
    path: str

    def run(self, config, job, data):
        vertices = data.get('vertices', None)
        displacements = data.get('displacements', None)
        faces = data.get('faces', None)

        with open(self.path, 'w') as f:
            f.write('Begin ModelPartData\n')
            f.write('End ModelPartData\n')
            f.write('\n')
            f.write('Begin Properties 0\n')
            f.write('End Properties\n')
            f.write('\n')
            f.write('Begin Nodes\n')
            for i, (x, y, z) in enumerate(vertices):
                f.write(f'  {i+1} {x} {y} {z}\n')
            f.write('End Nodes\n')
            f.write('\n')
            f.write('Begin Elements Element3D3N\n')
            for i, (a, b, c) in enumerate(faces):
                f.write(f'  {i+1} 0 {a+1} {b+1} {c+1}\n')
            f.write('End Elements\n')
            f.write('\n')
            f.write('Begin NodalData SHAPE_CHANGE_X\n')
            for i, u in enumerate(displacements[:, 0]):
                f.write(f'  {i+1} 0 {u}\n')
            f.write('End NodalData\n')
            f.write('\n')
            f.write('Begin NodalData SHAPE_CHANGE_Y\n')
            for i, u in enumerate(displacements[:, 1]):
                f.write(f'  {i+1} 0 {u}\n')
            f.write('End NodalData\n')
            f.write('\n')
            f.write('Begin NodalData SHAPE_CHANGE_Z\n')
            for i, u in enumerate(displacements[:, 2]):
                f.write(f'  {i+1} 0 {u}\n')
            f.write('End NodalData\n')


class ImportIbra(Task):
    path: str

    def run(self, config, job, data):
        model = an.Model()
        model.load(self.path)

        data['cad_model'] = model


class ExportIbra(Task):
    path: str

    def run(self, config, job, data):
        model = data.get('cad_model', None)

        if model is None:
            raise RuntimeError('No CAD model available')

        model.save(self.path)


