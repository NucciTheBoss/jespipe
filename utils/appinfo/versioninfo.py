import os


def versioninfo(name: str, year: str, show_w: str, show_c: str, *args, **kwargs) -> None:
    """
    Print out program version info in GNU GPLv3 format to standard output.

    ### Parameters:
    :param name: Name of program.
    :param year: Year of copyright.
    :param show_w: Content to place in `show_w` section.
    :param show_c: Content to place in `show_c` section.
    - args
      - author(s): Names of authors that hold copyright on program.
    - kwargs
      - ascii_banner: System file path to ASCII art banner to print out with version info.
    """
    # Print out banner
    banner_path = kwargs.get("ascii_banner") if "ascii_banner" in kwargs else None
    if banner_path is not None and os.path.isfile(banner_path):
        fin = open(banner_path, "rt"); banner = fin.read(); fin.close()
        print(banner, "\n")

    # Print program name, copyright year, and authors
    authors = ", ".join(args)
    print("{}  Copyright (C) {}  {}\n".format(name, year, authors))

    # Print warranty clause
    print("This program comes with ABSOLUTELY NO WARRANTY; for details please see the {} file.\n".format(show_w) +
    "This is free software, and you are welcome to redistribute it\n" +
    "under certain conditions; please visit {} for more details.".format(show_c))
