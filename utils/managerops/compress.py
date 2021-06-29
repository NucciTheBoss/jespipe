import tarfile
from zipfile import ZipFile
from . import getpaths as gp


class Compression:
    """A simple class that converts files into multiple data types."""
    def __init__(self, directory, name):
        """Compressor class initializer.
        
        Keyword arguments:
        directory -- directory to compress.
        name -- user-specified name to give created archive."""
        self.directory = directory
        self.name = name

    def tozip(self):
        """Compress files into a zip archive."""
        file_paths = gp.getfiles(self.directory)
        with ZipFile("{}.zip".format(self.name), "w") as zipfile:
            for file in file_paths:
                zipfile.write(file)

    def togzip(self):
        """Compress files into a tar.gz archive."""
        file_paths = gp.getfiles(self.directory)
        with tarfile.open("{}.tar.gz".format(self.name), "w:gz") as tarball:
            for file in file_paths:
                tarball.add(file)

    def tobzip(self):
        """Compress files into a tar.bz2 archive."""
        file_paths = gp.getfiles(self.directory)
        with tarfile.open("{}.tar.bz2".format(self.name), "w:bz2") as tarball:
            for file in file_paths:
                tarball.add(file)

    def toxz(self):
        """Compress files into a tar.xz archive."""
        file_paths = gp.getfiles(self.directory)
        with tarfile.open("{}.tar.xz".format(self.name), "w:xz") as tarball:
            for file in file_paths:
                tarball.add(file)

    def totar(self):
        """Compress files into a regular tar archive."""
        file_paths = gp.getfiles(self.directory)
        with tarfile.open("{}.tar".format(self.name), "w") as tarball:
            for file in file_paths:
                tarball.add(file)
