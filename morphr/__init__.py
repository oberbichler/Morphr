from morphr.configuration import Configuration, DebugData, Job, Task

from morphr.objectives.normal_distance import NormalDistance
from morphr.objectives.point_distance import PointDistance
from morphr.objectives.point_location import PointLocation
from morphr.objectives.point_node_coupling import PointNodeCoupling
from morphr.objectives.reduced_shell_3p import ReducedShell3P
from morphr.objectives.rotation_coupling import RotationCoupling
from morphr.objectives.shell_3p import Shell3P

from morphr.logging import Logger

from morphr.tasks.apply_alpha_regularization import ApplyAlphaRegularization
from morphr.tasks.apply_edge_coupling import ApplyEdgeCoupling
from morphr.tasks.apply_mesh_displacement import ApplyMeshDisplacement
from morphr.tasks.apply_reduced_shell_3p import ApplyReducedShell3P
from morphr.tasks.apply_shell_3p import ApplyShell3P
from morphr.tasks.export_ibra import ExportIbra
from morphr.tasks.export_mdpa import ExportMdpa
from morphr.tasks.import_displacement_field import ImportDisplacementField
from morphr.tasks.import_ibra import ImportIbra
from morphr.tasks.solve_nonlinear import SolveNonlinear


__version__ = 'dev'

__all__ = [
    'Configuration',
    'DebugData',
    'Job',
    'Logger',
    'Task',
    # objectives
    'NormalDistance',
    'PointDistance',
    'PointLocation',
    'PointNodeCoupling',
    'ReducedShell3P',
    'RotationCoupling',
    'Shell3P',
    # tasks
    'ApplyAlphaRegularization',
    'ApplyEdgeCoupling',
    'ApplyMeshDisplacement',
    'ApplyReducedShell3P',
    'ApplyShell3P',
    'ExportIbra',
    'ExportMdpa',
    'ImportDisplacementField',
    'ImportIbra',
    'SolveNonlinear',
]
