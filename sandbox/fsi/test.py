from dolfin import *

mesh = UnitSquare(2, 2)
cell = Cell(mesh, 0)

print cell.num_entities(0), len(cell.entities(0))
