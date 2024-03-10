import glob
import numpy as np

file = 0 # 0 for 1000_euclidianDistance.txt and 1 for 1000_randomDistance.txt



# Get path for all text files in current directory
text_files = glob.glob('data/*.txt')
print(text_files)

with open(text_files[1], 'r') as f:
    N = f.readline()    # Number of nodes
    print(f'Number of nodes: {N}')
    edges = np.zeros((int(N), int(N)))

    next(f) # Skip the next line
    for line in f:
        n1, n2, d = line.split()
        edges[int(n1)-1, int(n2)-1] = d
        edges[int(n2)-1, int(n1)-1] = d

print(edges)


import time
from tqdm import tqdm
from multiprocessing import Process
from Lin_Kernighan import LKAlgorithm

def fun(LK: LKAlgorithm):
    tour, dist = LK.optimize()
    print(f"Best path has length {dist}")
    print(f"Best path is {tour}")

if __name__ == '__main__':
    LK = LKAlgorithm(edges)
    p = Process(target=fun, args=(LK,))

    p.start()
    # Sleep for 15 minutes
    with tqdm(total=15, desc="Running LK Algorithm") as pbar:
        for i in range(15):
            time.sleep(60)
            pbar.update(1)

    p.terminate()
    p.join()

# print(f"Best path has length {dist}")
# print(f"Best path is {tour}")