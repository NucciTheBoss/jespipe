import tarfile
from zipfile import ZipFile

from tqdm import tqdm

from ..filesystem import getpaths as gp


class Compression:
    def __init__(self, directory: str, name: str) -> None:
        """
        Compressor class to facilitate compressing directories
        into into either bzip, gzip, tar, xz, or zip format.
        Saves archive as either tar.bz2, tar.gz, .tar, tar.xz, or .zip.
        
        ### Parameters:
        :param directory: System file path of directory to compress into an archive.
        :param name: User-specified name to use for archive.

        ### Methods:
        - public
          - tobzip: Compress directory into a tar.bz2 archive.
          - togzip: Compress directory into a tar.gz archive.
          - totar: Compress directory into a regular tar archive.
          - toxz: Compress directory into a tar.xz archive.
          - tozip: Compress directory into a zip archive.
        """
        self.directory = directory
        self.name = name

    def tobzip(self, **kwargs) -> None:
        """
        Compress directory into a tar.bz2 archive.
        Saves archive as {self.name}.tar.bz2.

        ### Parameters:
        - kwargs
          - verbose: Display progress bar. (default: True)
        """
        verbose = True if kwargs.get("verbose") is None else kwargs.get("verbose")
        if isinstance(verbose, bool) is False:
            verbose = True

        file_paths = gp.getfiles(self.directory)
        with tarfile.open("{}.tar.bz2".format(self.name), "w:bz2") as tarball:
            for file in tqdm(file_paths, desc="bzip2 completion progress", disable=not verbose):
                tarball.add(file)

    def togzip(self, **kwargs) -> None:
        """
        Compress directory into a tar.gz archive.
        Saves archive as {self.name}.tar.gz.

        ### Parameters:
        - kwargs
          - verbose: Display progress bar. (default: True)
        """
        verbose = True if kwargs.get("verbose") is None else kwargs.get("verbose")
        if isinstance(verbose, bool) is False:
            verbose = True

        file_paths = gp.getfiles(self.directory)
        with tarfile.open("{}.tar.gz".format(self.name), "w:gz") as tarball:
            for file in tqdm(file_paths, desc="gzip completion progress", disable=not verbose):
                tarball.add(file)

    def totar(self, **kwargs) -> None:
        """
        Compress directory into a regular tar archive.
        Saves archive as {self.name}.tar.

        ### Parameters:
        - kwargs
          - verbose: Display progress bar. (default: True)
        """
        verbose = True if kwargs.get("verbose") is None else kwargs.get("verbose")
        if isinstance(verbose, bool) is False:
            verbose = True

        file_paths = gp.getfiles(self.directory)
        with tarfile.open("{}.tar".format(self.name), "w") as tarball:
            for file in tqdm(file_paths, desc="tar completion progress", disable=not verbose):
                tarball.add(file)

    def toxz(self, **kwargs) -> None:
        """
        Compress directory into a tar.xz archive.
        Saves archive as {self.name}.tar.xz.

        ### Parameters:
        - kwargs
          - verbose: Display progress bar. (default: True)
        """
        verbose = True if kwargs.get("verbose") is None else kwargs.get("verbose")
        if isinstance(verbose, bool) is False:
            verbose = True

        file_paths = gp.getfiles(self.directory)
        with tarfile.open("{}.tar.xz".format(self.name), "w:xz") as tarball:
            for file in tqdm(file_paths, desc="xz completion progress", disable=not verbose):
                tarball.add(file)

    def tozip(self, **kwargs) -> None:
        """
        Compress directory into a zip archive.
        Saves archive as {self.name}.zip.

        ### Parameters:
        - kwargs
          - verbose: Display progress bar. (default: True)
        """
        verbose = True if kwargs.get("verbose") is None else kwargs.get("verbose")
        if isinstance(verbose, bool) is False:
            verbose = True

        file_paths = gp.getfiles(self.directory)
        with ZipFile("{}.zip".format(self.name), "w") as zipfile:
            for file in tqdm(file_paths, desc="zip completion progress", disable=not verbose):
                zipfile.write(file)
