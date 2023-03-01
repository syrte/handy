# handy
Some handy python functions for statistics, computation and plotting.

## Document
See the docstring of each function.
It is pretty good in general.

## Install
Just clone 'handy' to a directory on your computer. I used to use put `handy` at `~/lib/handy`. 
```bash
cd ~/lib/
git clone https://github.com/syrte/handy.git
```

Optionally, you can set the enviroment variable, so that Python knows where to find it hereafter.
```
echo "export PYTHONPATH=$HOME/lib:$PYTHONPATH" >> ~/.bashrc
```

I might consider pushing it to pypi.org in the future.

## Usage
If you have set `PYTHONPATH` in .bashrc as above, then you can simply `import handy` in Python
```
import handy
```

Otherwise, you should tell Python where to find it before importing
```
import sys
sys.path.append('path_dir/')

import handy
```
Note that `path_dir` should the the absolute path of the *parent* directory where you placed 'handy',
in my case, it is `/home/username/lib/`.


## Update

```bash
cd ~/lib/handy
git pull
```
