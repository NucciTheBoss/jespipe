def licenseinfo(desc: str, year: str, *args) -> None:
    """
    Print out program license info in GNU GPLv3 format to standard output.

    ### Parameters:
    :param desc: Name and short description of program.
    :param year: Year of copyright.
    - args
      - author(s): Names of authors that hold copyright on program.
    """
    # Print out program description
    print(desc, "\n")

    # Print out copyright year and authors
    authors = ", ".join(args)
    print("Copyright (C) {}  {}\n".format(year, authors))

    # Print out license info
    print("This program is free software: you can redistribute it and/or modify\n" +
    "it under the terms of the GNU General Public License as published by\n" +
    "the Free Software Foundation, either version 3 of the License, or\n" +
    "(at your option) any later version.\n\n" +

    "This program is distributed in the hope that it will be useful,\n" +
    "but WITHOUT ANY WARRANTY; without even the implied warranty of\n" +
    "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\n" +
    "GNU General Public License for more details.\n\n" +

    "You should have received a copy of the GNU General Public License\n" +
    "along with this program.  If not, see <https://www.gnu.org/licenses/>.\n")
