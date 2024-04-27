import numpy as np

def read(file: str):
    result = []
    with open(file) as f:
        for line in f.readlines():
            result.append(list(map(float, line.split())))

    return np.array(result)

def write(file: str, matrix):
    rows, cols = matrix.shape
    with open(file, "w") as f:
        for i in range(rows):
            for j in range(cols):
                f.write(f"{matrix[i][j]} ")
            f.write("\n")


A = read("A.matrix")
B = read("B.matrix")
C = read("C.matrix")

print(f"A.mean: {A.mean()}")
print(f"B.mean: {B.mean()}")
print(f"C.mean: {C.mean()}")

C2 = np.matmul(A, B);

D = C2 - C

write("C2.matrix", C2)
write("DeltaC.matrix", D)



rows, cols = D.shape

for i in range(rows):
    for j in range(cols):
        if D[i][j] > 0.1:
            print(D[i][j])
