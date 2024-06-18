# HPS SIMPy
Shared python snippets for HPS analysis
mainly focused on my work searching for SIMPs.

## Usage
Since these python snippets are constantly evolving along
with the analysis, they are not installed. Instead, I just
clone this repository into the working directory where my
scripts or notebooks are so that this is available without
modification of `PYTHONPATH`.
```
git clone --recursive git@github.com:tomeichlersmith/hps-simpy.git simpy
```
Then, in a script or notebook whose current directory is the directory
in which we ran the above command, we can import the various modules
stored in this repository.
```python
import simpy
from simpy.plot import plt, histplot
# etc...
```
If you find yourself editing these files while using them within a notebook, the 
[autoreload](https://ipython.readthedocs.io/en/stable/config/extensions/autoreload.html)
extension is helpful for making the modules reload if the source files have changed.
```
%load_ext autoreload
%autoreload 2
```
**Note**: The `exclusion` submodule here is its own set of python modules
similar to this set. There exists some repeated code and I am not interested
in aligning the two, so just be careful.