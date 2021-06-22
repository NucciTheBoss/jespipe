#%Module 1.0
#
#   Python 3.9.5 module to be used with jespipe on Ursula:
#
set                 home_env_var            $::env(HOME)
set                 root_path               "$home_env_var/sw/python-3.9.5"
prepend-path        PATH                    "$root_path/bin"
prepend-path        LD_LIBRARY_PATH         "$root_path/lib"
prepend-path        LIBRARY_PATH            "$root_path/lib"
prepend-path        C_INCLUDE_PATH          "$root_path/include"
prepend-path        CPLUS_INCLUDE_PATH      "$root_path/include"
prepend-path        INCLUDE                 "$root_path/include"
prepend-path        PKG_CONFIG_PATH         "$root_path/lib/pkgconfig"
prepend-path        MANPATH                 "$root_path/share/man"
prepend-path        CMAKE_PREFIX_PATH       "$root_path"
