# %%
import os
from numba import jit, prange
from pprint import pprint
# initialize the generator for the respective folders
# ==============================================================================
training_path =  os.walk(os.path.join("dataset", "training"), topdown=False)
validation_gallery_path =  os.walk(os.path.join("dataset", "validation", "gallery"), topdown=False)
validation_query_path =  os.walk(os.path.join("dataset", "validation", "query"), topdown=False)

# TODO inventarsi un nome migliore per la classe
class Item:
    def __init__(self, mygenerator):
        self.list_dirs = []
        self.list_files = []
        self.mygenerator = mygenerator
        for root, dirs, files in self.mygenerator:
            for d in prange(len(dirs)):
                self.list_dirs.append(os.path.join(root, dirs[d]))
            for f in  prange(len(files)):
                self.list_files.append(os.path.join(root, files[f]))

    def print_dirs(self):
        pprint(self.list_dirs)
    def get_dirs(self):
        return self.list_dirs
    
    def print_files(self):
        pprint(self.list_files)
    def get_files(self):
        return self.list_files
# %%


Training = Item(training_path)
Validation_Galery = Item(validation_gallery_path)
Validation_Query = Item(validation_query_path)


# %%

# Debug pprint all files
# ==============================================================================
Training.print_files()
Validation_Galery.print_files()
Validation_Query.print_files()
# Debug pprint all files
# ==============================================================================
Training.print_dirs()
Validation_Galery.print_dirs()
Validation_Query.print_dirs()
# %%
