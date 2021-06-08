help([[Python is an interpreted high-level general-purpose programming 
language. Python's design philosophy emphasizes code readability with its 
notable use of significant indentation. Its language constructs as well as 
its object-oriented approach aim to help programmers write clear, 
logical code for small and large-scale projects.]])

-- General information
whatis("Description: Python is a programming language that lets you work quickly and integrate systems more effectively.")
whatis("Python Version: 3.8.8")
whatis("Compile Options: --enable-shared --enable-optimizations --prefix=$HOME/python-3.8.8")

-- Pull $HOME and concatenate paths to python-3.8.8
local home = os.getenv("HOME")
local root = string.format("%s/python-3.8.8", home)

local pythonpath = pathJoin(root, "bin")
local pythonlib = pathJoin(root, "lib")
local pythoninc = pathJoin(root, "include")
local pythonman = pathJoin(root, "share/man")
local pythonpkgconfig = pathJoin(root, "lib/pkgconfig")

-- Prepend paths
prepend_path("PATH", pythonpath)
prepend_path("LIBRARY_PATH", pythonlib)
prepend_path("LD_LIBRARY_PATH", pythonlib)
prepend_path("INCLUDE", pythoninc)
prepend_path("C_INCLUDE_PATH", pythoninc)
prepend_path("CPLUS_INCLUDE_PATH", pythoninc)
prepend_path("MANPATH", pythonman)
prepend_path("PKG_CONFIG_PATH", pythonpkgconfig)
prepend_path("CMAKE_PREFIX_PATH", root)
