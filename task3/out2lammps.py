#!/usr/bin/python3

from sys import stdin

def print_lammps(step, atoms):
    print(f"ITEM: TIMESTEP\n{step}")
    print(f"ITEM: NUMBER OF ATOMS\n{len(atoms)}")
    print(f"ITEM: BOX BOUNDS ff ff ff\n0 16\n0 16\n0 16")
    print(f"ITEM: ATOMS id x y z")

    i = 1
    for atom in atoms:
        print(f"{i} {atom[1]} {atom[2]} {atom[3]}")
        i += 1

temp = []
step = 0
for line in stdin:
    if line == "0 0 0 0\n":
        print_lammps(step, temp)
        temp = []
        step += 1
    else:
        temp.append(tuple(map(int, line.split())))

