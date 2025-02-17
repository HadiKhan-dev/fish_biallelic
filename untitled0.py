import sys
import resource
import multiprocess
#%%
from memory_profiler import profile
#%%
resource.setrlimit(resource.RLIMIT_AS,(10**5, 10**5))
#%%
print((resource.getrlimit(resource.RLIMIT_AS)))
#%%
print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
#%%
print(sys.getsizeof(memory_hog))
#%%
memory_hog = {}
try:
    for x in range(10000):
        print(x)
        memory_hog[str(x)]='The sky is so bued'
except MemoryError as err:
    #display(err)
    sys.exit('memory exceeded')
    # memory exceeded
    pass
