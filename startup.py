print('Running .startup')
try:
    import cPickle as pickle  # Python 2
except:
    import pickle   # Python 3
import os
import sys
from os.path import expanduser

home = expanduser("~")
local_repo = os.path.join("Documents", "Code", "qfl")
os.chdir(os.path.join(home, local_repo))  # Activate .env

home = os.path.expanduser("~")
local_repo = os.path.join("Documents", "Code", "qfl")
sys.path.append(os.path.join(home, local_repo))  # Activate .env

modules = ["qfl"]
sub_modules =['qfl', 'airflow']
for sm in sub_modules:
    modules.append(os.path.join(modules[0], sm))
modules.append(os.path.join("qfl", "etl"))
modules.append(os.path.join("qfl", "core"))
modules.append(os.path.join("airflow", "dags"))

modules.append(os.path.join('data_extraction', 'database'))
sys.path.extend([os.path.join(home, local_repo, p) for p in modules])

print('startup complete!')
