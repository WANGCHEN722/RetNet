import torch

def preallocate(*shape, seed=0):
    torch.manual_seed(seed)
    x = torch.empty(shape)
    for i in range(shape[0]):
        torch.rand(shape[1:], out=x[i])

    return x


def stacker(*shape, seed=0):
    torch.manual_seed(seed)
    x = []
    for _ in range(shape[0]):
        x.append(torch.rand(shape[1:]))

    return torch.stack(x)


if __name__ == "__main__":
    print("Testing...")
    import timeit
    import random

    seed = random.randint(-0x8000_0000_0000_0000, 0xffff_ffff_ffff_ffff)
    print(timeit.timeit(f"preallocate(1000, seed={seed})",
          "from __main__ import preallocate", number=1000))
    print(timeit.timeit(f"stacker(1000, seed={seed})",
          "from __main__ import stacker", number=1000))
    
    data=[random.random() for _ in range(10000)]
    print(timeit.timeit(f"statistics.mean({data})", "import statistics", number=10000))
    print(timeit.timeit(f"statistics.fmean({data})", "import statistics", number=10000))
    print(timeit.timeit(f"sum({data})/len({data})", number=10000))
