from morphr.configuration import Configuration, DebugData, Job, Task

from morphr.objectives.iga_normal_distance_ad import IgaNormalDistanceAD
from morphr.objectives.iga_point_distance_ad import IgaPointDistanceAD
from morphr.objectives.iga_point_location_ad import IgaPointLocationAD
from morphr.objectives.iga_point_node_coupling_ad import IgaPointNodeCouplingAD
from morphr.objectives.iga_rotation_coupling_ad import IgaRotationCouplingAD
from morphr.objectives.iga_shell_3p_ad import IgaShell3PAD
from morphr.objectives.reduced_shell import ReducedIgaShell

from morphr.logging import Logger

from morphr.tasks.apply_alpha_regularization import ApplyAlphaRegularization
from morphr.tasks.apply_edge_coupling import ApplyEdgeCoupling
from morphr.tasks.apply_membrane_3p import ApplyMembrane3P
from morphr.tasks.apply_mesh_displacement import ApplyMeshDisplacement
from morphr.tasks.apply_reduced_shell import ApplyReducedShell
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
    'IgaNormalDistanceAD',
    'IgaPointDistanceAD',
    'IgaPointLocationAD',
    'IgaPointNodeCouplingAD',
    'ReducedIgaShell',
    'IgaRotationCouplingAD',
    'IgaShell3PAD',
    # tasks
    'ApplyAlphaRegularization',
    'ApplyEdgeCoupling',
    'ApplyMembrane3P',
    'ApplyMeshDisplacement',
    'ApplyReducedShell',
    'ApplyShell3P',
    'ExportIbra',
    'ExportMdpa',
    'ImportDisplacementField',
    'ImportIbra',
    'SolveNonlinear',
]
