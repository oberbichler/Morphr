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
from morphr.conditions import DisplacementCoupling, PointOnSurfaceSupport, Shell3P3D, EdgeRotationCoupling, PointOnSurfaceNodeCoupling


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

        if 'triangle' in mesh_0.cells_dict:
            for a, b, c in mesh_0.cells_dict['triangle']:
                faces.append([a, b, c])

        if 'quad' in mesh_0.cells_dict:
            for a, b, c, d in mesh_0.cells_dict['quad']:
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


class SolveNonlinear(Task):
    max_iterations: int = 100
    damping: float = 0

    def run(self, config, job, data):
        elements = data.get('elements', None)

        problem = eq.Problem(elements, nb_threads=1)

        eq.Log.info_level = 5

        solver = eq.NewtonRaphson(problem)
        solver.maxiter = self.max_iterations
        solver.damping = self.damping

        solver.run()

        for surface, nodes in data['nodes'].items():
            for i, node in enumerate(nodes):
                surface.set_pole(i, node.act_location)


class ApplyMeshDisplacement(Task):
    penalty: float = 1
    debug: bool = False

    def line_projection(self, point, a, b):
        dif = b - a
        dot = np.dot(dif, dif)

        if dot < 1e-14:
            return a, 0.0

        o = a
        r = dif / dot
        o2pt = point - o
        t = np.dot(o2pt, r)

        if t < 0:
            return a, 0.0

        if t > 1:
            return b, 1.0

        closest_point = o + dif * t

        return closest_point, t

    def run(self, config, job, data):
        cad_model = data.get('cad_model', None)
        vertices = data.get('vertices', None)
        displacements = data.get('displacements', None)
        faces = data.get('faces', None)
        penalty = self.penalty
        model_tolerance = job.model_tolerance

        # FIXME: Check for None

        data['nodes'] = data.get('nodes', {})

        data['elements'] = elements = data.get('elements', [])

        rtree = an.RTree3D(len(faces))

        for face in faces:
            vabc = vertices[face]

            box_min = np.min(vabc, axis=0)
            box_max = np.max(vabc, axis=0)

            if self.debug:
                cad_model.add(an.Box3D(box_min, box_max), r'{"layer": "Debug/ApplyMeshDisplacement/Boxes"}')

            rtree.add(box_min, box_max)

        rtree.finish()

        for key, face in cad_model.of_type('BrepFace'):
            surface_geometry_key = surface_geometry = face.surface_geometry.data

            span_nodes = dict()
            span_data = dict()

            if surface_geometry_key not in data['nodes']:
                nodes = []

                for x, y, z in surface_geometry.poles:
                    nodes.append(eq.Node(x, y, z))

                nodes = np.array(nodes, object)

                data['nodes'][surface_geometry_key] = nodes
            else:
                nodes = data['nodes'][surface_geometry_key]

            for u, v, weight in an.integration_points(face, model_tolerance):
                location = surface_geometry.point_at(u, v)

                if self.debug:
                    cad_model.add(an.Point3D(location), r'{"layer": "Debug/ApplyMeshDisplacement/IntegrationPoints"}')

                indices = rtree.by_point(location, model_tolerance)

                min_distance2 = float('inf')
                min_location = None
                min_parameter = None
                min_face = None

                for index in indices:
                    a, b, c = faces[index]

                    va, vb, vc = vertices[[a, b, c]]

                    closest_point, parameter = an.Triangle3D.projection(location, va, vb, vc)

                    if np.min(parameter) < 0 or np.max(parameter) > 1:
                        continue

                    d = closest_point - location
                    distance2 = d.dot(d)

                    if distance2 > min_distance2:
                        continue

                    min_distance2 = distance2
                    min_location = closest_point
                    min_parameter = parameter
                    min_face = index

                    if distance2 < model_tolerance**2:
                        break

                if min_face is None:
                    continue

                abc = faces[min_face]

                dabc = displacements[abc]

                displacement = min_parameter.dot(dabc)

                nonzero_indices, shape_functions = surface_geometry.shape_functions_at(u, v, 0)

                location_source = min_location
                location_target = min_location + displacement

                element_data = [shape_functions, min_location - displacement, weight * penalty]

                span_u = an.upper_span(surface_geometry.degree_u, surface_geometry.knots_u, u)
                span_v = an.upper_span(surface_geometry.degree_v, surface_geometry.knots_v, v)

                span = (span_u, span_v)

                old_data = span_data.get(span, [])

                if len(old_data) == 0:
                    span_nodes[span] = nodes[nonzero_indices]

                span_data[span] = old_data + [element_data]

                if self.debug:
                    cad_model.add(an.Point3D(location_source), r'{"layer": "Debug/ApplyMeshDisplacement/ClosestPoints"}')
                    cad_model.add(an.Line3D(location_source, location_target), r'{"layer": "Debug/ApplyMeshDisplacement/DisplacementFields"}')
                    cad_model.add(an.Line3D(location, location_source), r'{"layer": "Debug/ApplyMeshDisplacement/Projections"}')

                element = PointOnSurfaceSupport(nodes[nonzero_indices], shape_functions, min_location + displacement, weight * penalty)
                elements.append(element)


class ApplyShell3P3D(Task):
    thickness: float
    youngs_modulus: float
    poissons_ratio: float

    def run(self, config, job, data):
        cad_model = data.get('cad_model', None)
        model_tolerance = job.model_tolerance

        # FIXME: Check for None

        data['nodes'] = data.get('nodes', {})

        data['elements'] = elements = data.get('elements', [])

        thickness = self.thickness
        youngs_modulus = self.youngs_modulus
        poissons_ratio = self.poissons_ratio

        for key, face in cad_model.of_type('BrepFace'):
            surface_geometry_key = surface_geometry = face.surface_geometry.data

            if surface_geometry_key not in data['nodes']:
                nodes = []

                for x, y, z in surface_geometry.poles:
                    nodes.append(eq.Node(x, y, z))
                nodes = np.array(nodes, object)
                data['nodes'][surface_geometry_key] = nodes
            else:
                nodes = data['nodes'][surface_geometry_key]

            for u, v, weight in an.integration_points(face, model_tolerance):
                nonzero_indices, shape_functions = surface_geometry.shape_functions_at(u, v, 2)

                element = Shell3P3D(nodes[nonzero_indices], shape_functions, thickness, youngs_modulus, poissons_ratio, weight)
                elements.append(element)


class ApplyEdgeCoupling(Task):
    penalty_displacement: float = 1.0
    penalty_rotation: float = 1.0
    debug: bool = False

    def run(self, config, job, data):
        model_tolerance = job.model_tolerance
        cad_model = data.get('cad_model', None)
        penalty_displacement = self.penalty_displacement
        penalty_rotation = self.penalty_rotation

        # FIXME: Check for None

        data['nodes'] = nodes = data.get('nodes', {})
        data['elements'] = elements = data.get('elements', [])

        for key, edge in cad_model.of_type('BrepEdge'):
            if edge.nb_trims != 2:
                continue

            (_, trim_a), (_, trim_b) = edge.trims

            nurbs_surface_key_a, nurbs_surface_a = trim_a.surface_geometry
            nurbs_surface_key_b, nurbs_surface_b = trim_b.surface_geometry

            if nurbs_surface_a not in nodes:
                nurbs_surface_nodes = []

                for x, y, z in nurbs_surface_a.poles:
                    nurbs_surface_nodes.append(eq.Node(x, y, z))

                nurbs_surface_nodes_a = np.array(nurbs_surface_nodes, object)

                nodes[nurbs_surface_a] = nurbs_surface_nodes
            else:
                nurbs_surface_nodes_a = nodes[nurbs_surface_a]

            if nurbs_surface_b not in nodes:
                nurbs_surface_nodes = []

                for x, y, z in nurbs_surface_b.poles:
                    nurbs_surface_nodes.append(eq.Node(x, y, z))

                nurbs_surface_nodes_b = np.array(nurbs_surface_nodes, object)

                nodes[nurbs_surface_b] = nurbs_surface_nodes
            else:
                nurbs_surface_nodes_b = nodes[nurbs_surface_b]

            integration_points_a, integration_points_b = an.integration_points(edge, tolerance=model_tolerance)

            for (t_a, weight), (t_b, _) in zip(integration_points_a, integration_points_b):
                u_a, v_a = trim_a.curve_geometry.data.point_at(t_a)
                u_b, v_b = trim_b.curve_geometry.data.point_at(t_b)

                indices_a, shape_functions_a = nurbs_surface_a.shape_functions_at(u_a, v_a, 1)
                indices_b, shape_functions_b = nurbs_surface_b.shape_functions_at(u_b, v_b, 1)

                element_nodes_a = [nurbs_surface_nodes_a[i] for i in indices_a]
                element_nodes_b = [nurbs_surface_nodes_b[i] for i in indices_b]

                if penalty_displacement != 0:
                    element = DisplacementCoupling(element_nodes_a, element_nodes_b, shape_functions_a, shape_functions_b, weight * penalty_displacement)
                    elements.append(element)

                if penalty_rotation != 0:
                    _, t2_edge = trim_a.curve_3d.derivatives_at(t_a, order=1)
                    t2_edge /= np.linalg.norm(t2_edge)

                    element = EdgeRotationCoupling(element_nodes_a, element_nodes_b, shape_functions_a, shape_functions_b, t2_edge, weight * penalty_rotation)
                    elements.append(element)

                    if self.debug:
                        cad_model.add(an.Point3D(element.act_b), r'{"layer": "Debug/ApplyEdgeCoupling/RotationAxis"}')

                if self.debug:
                    point_a = nurbs_surface_a.point_at(u_a, v_a)
                    point_b = nurbs_surface_b.point_at(u_b, v_b)
                    cad_model.add(an.Point3D(point_a), r'{"layer": "Debug/ApplyEdgeCoupling/PointsA"}')
                    cad_model.add(an.Point3D(point_b), r'{"layer": "Debug/ApplyEdgeCoupling/PointsB"}')

                    if penalty_rotation != 0:
                        cad_model.add(an.Line3D(point_a, point_a + t2_edge), r'{"layer": "Debug/ApplyEdgeCoupling/RotationAxis"}')


class ApplyAlphaRegularization(Task):
    penalty: float = 1.0
    debug: bool = False

    def run(self, config, job, data):
        cad_model = data.get('cad_model', None)

        # FIXME: Check for None

        data['nodes'] = data.get('nodes', {})

        data['elements'] = elements = data.get('elements', [])

        nb_conditions = 0

        for key, face in cad_model.of_type('BrepFace'):
            surface_geometry_key = surface_geometry = face.surface_geometry.data

            if surface_geometry_key not in data['nodes']:
                nodes = []

                for x, y, z in surface_geometry.poles:
                    nodes.append(eq.Node(x, y, z))
                nodes = np.array(nodes, object)
                data['nodes'][surface_geometry_key] = nodes
            else:
                nodes = data['nodes'][surface_geometry_key]

            for r in range(surface_geometry.nb_poles_u):
                for s in range(surface_geometry.nb_poles_v):
                    target_node = nodes[r * surface_geometry.nb_poles_v + s]
                    u, v = surface_geometry.greville_point(r, s)

                    nonzero_indices, shape_functions = surface_geometry.shape_functions_at(u, v, 0)

                    element = PointOnSurfaceNodeCoupling(nodes[nonzero_indices], shape_functions, target_node, self.penalty)
                    elements.append(element)

                    nb_conditions += 1

                    if self.debug:
                        cad_model.add(an.Line3D(target_node.act_location, surface_geometry.point_at(u, v)), r'{"layer": "Debug/ApplyAlphaRegularization/Connections"}')

        print(f'{nb_conditions} new conditions')
