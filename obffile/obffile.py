# obffile.py

# Copyright (c) 2025-2026, Christoph Gohlke
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Read Imspector object binary format files (OBF and MSR).

Obffile is a Python library to read image and metadata from
Object Binary Format (OBF) and Measurement Summary Record (MSR) image files.
These files are written by Imspector software to store image and metadata
from microscopy experiments.

:Author: `Christoph Gohlke <https://www.cgohlke.com>`_
:License: BSD-3-Clause
:Version: 2026.2.20

Quickstart
----------

Install the obffile package and all dependencies from the
`Python Package Index <https://pypi.org/project/obffile/>`_::

    python -m pip install -U obffile[all]

See `Examples`_ for using the programming interface.

Source code and support are available on
`GitHub <https://github.com/cgohlke/obffile>`_.

Requirements
------------

This revision was tested with the following requirements and dependencies
(other versions may work):

- `CPython <https://www.python.org>`_ 3.11.9, 3.12.10, 3.13.12, 3.14.3 64-bit
- `NumPy <https://pypi.org/project/numpy>`_ 2.4.2
- `Xarray <https://pypi.org/project/xarray>`_ 2026.2.0 (recommended)
- `Matplotlib <https://pypi.org/project/matplotlib/>`_ 3.10.8 (optional)
- `Tifffile <https://pypi.org/project/tifffile/>`_ 2026.2.16 (optional)

Revisions
---------

2026.2.20

- Initial alpha release.
- …

Notes
-----

`Imspector <https://imspectordocs.readthedocs.io>`_ is a software platform for
super-resolution and confocal microscopy developed by Abberior Instruments.

This library is in its early stages of development. It is not feature-complete.
Large, backwards-incompatible changes may occur between revisions.

Specifically, the following features are not supported:
writing or modifying OBF/MSR files, non-OBF based MSR files, reading
MSR-specific non-image data (window positions, hardware configuration),
and compression types other than zlib.

The library has been tested with a limited number of files only.

The Imspector image file formats are documented at
https://imspectordocs.readthedocs.io/en/latest/fileformat.html.

Other implementations for reading Imspector image files are
`msr-reader <https://github.com/hoerlteam/msr-reader>`_,
`obf_support.py <https://github.com/biosciflo/VISION>`_, and
`bio-formats <https://github.com/ome/bioformats>`_.

Examples
--------

Read an image stack and metadata from a OBF file:

>>> with ObfFile('tests/data/Test.obf') as obf:
...     assert obf.header.metadata['ome_xml'].startswith('<?xml')
...     for stack in obf.stacks:
...         _ = stack.name, stack.dims, stack.shape, stack.dtype
...     obf.stacks[0].asxarray()
...
<xarray.DataArray 'Abberior STAR RED.Confocal' (T: 18, Z: 2, Y: 339, X: 381)...
array([[[[0, 0, 0, ..., 3, 3, 3],
         [0, 0, 0, ..., 4, 3, 3],
         ...,
         [0, 0, 0, ..., 0, 0, 0],
         [0, 0, 0, ..., 0, 0, 0]]]], shape=(18, 2, 339, 381), dtype=int16)
Coordinates:
    * T        (T) float64 144B 0.0 0.0 0.0 0.0 0.0 0.0 ...
    * Z        (Z) float64 16B 1.25e-07 3.75e-07
    * Y        (Y) float64 3kB 0.0 2e-07 4e-07 ...
    * X        (X) float64 3kB 0.0 2.002e-07 4.003e-07 ...
...

View the image stack and metadata in a OBF file from the console::

    $ python -m obffile tests/data/Test.obf

"""

from __future__ import annotations

__version__ = '2026.2.20'

__all__ = [
    'FILE_EXTENSIONS',
    'ObfFile',
    'ObfFileError',
    'ObfFileHeader',
    'ObfSiUnit',
    'ObfStack',
    'ObfStackFooter',
    'ObfStackHeader',
    'ObfStackSequence',
    '__version__',
    'imread',
]

import contextlib
import dataclasses
import io
import math
import os
import re
import struct
import sys
import zlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, ClassVar, final, overload

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import TracebackType
    from typing import IO, Any, Literal, Self

    from numpy.typing import NDArray
    from xarray import DataArray

import numpy


@overload
def imread(
    file: str | os.PathLike[Any] | IO[bytes],
    /,
    stack: int = 0,
    *,
    squeeze: bool = True,
    asxarray: Literal[False] = ...,
) -> NDArray[Any]: ...


@overload
def imread(
    file: str | os.PathLike[Any] | IO[bytes],
    /,
    stack: int = 0,
    *,
    squeeze: bool = True,
    asxarray: Literal[True] = ...,
) -> DataArray: ...


def imread(
    file: str | os.PathLike[Any] | IO[bytes],
    /,
    stack: int = 0,
    *,
    squeeze: bool = True,
    asxarray: bool = False,
) -> NDArray[Any] | DataArray:
    """Return image stack from Object Binary Format file.

    Dimensions are returned in order stored in file.

    Parameters:
        file:
            Name of Object Binary Format file or seekable binary stream.
        stack:
            Index of image stack to read.
        squeeze:
            Remove dimensions of length one from stacks.
        asxarray:
            Return image data as xarray.DataArray instead of numpy.ndarray.

    Returns:
        :
            Image stack data as numpy array or xarray DataArray.

    """
    with ObfFile(file, squeeze=squeeze) as obf:
        im = obf.stacks[stack]
        return im.asxarray() if asxarray else im.asarray()


class ObfFileError(ValueError):
    """Exception to indicate invalid Object Binary Format file structure."""


class BinaryFile:
    """Binary file.

    Parameters:
        file:
            File name or seekable binary stream.
        mode:
            File open mode if `file` is a file name.
            If not specified, defaults to 'r'. Files are always opened
            in binary mode.

    Raises:
        ValueError:
            Invalid file name, extension, or stream.
            File is not a binary or seekable stream.

    """

    _fh: IO[bytes]
    _path: str  # absolute path of file
    _name: str  # name of file or handle
    _close: bool  # file needs to be closed
    _closed: bool  # file is closed
    _ext: ClassVar[set[str]] = set()  # valid extensions, empty for any

    def __init__(
        self,
        file: str | os.PathLike[str] | IO[bytes],
        /,
        *,
        mode: Literal['r', 'r+'] | None = None,
    ) -> None:

        self._path = ''
        self._name = 'Unnamed'
        self._close = False
        self._closed = False

        if isinstance(file, (str, os.PathLike)):
            ext = os.path.splitext(file)[-1].lower()
            if self._ext and ext not in self._ext:
                msg = f'invalid file extension: {ext!r} not in {self._ext!r}'
                raise ValueError(msg)
            if mode is None:
                mode = 'r'
            else:
                if mode[-1:] == 'b':
                    mode = mode[:-1]  # type: ignore[assignment]
                if mode not in {'r', 'r+'}:
                    msg = f'invalid {mode=!r}'
                    raise ValueError(msg)
            self._path = os.path.abspath(file)
            self._close = True
            self._fh = open(self._path, mode + 'b')  # noqa: SIM115

        elif hasattr(file, 'seek'):
            # binary stream: open file, BytesIO, fsspec LocalFileOpener
            if isinstance(file, io.TextIOBase):  # type: ignore[unreachable]
                msg = f'{file=!r} is not open in binary mode'
                raise TypeError(msg)

            self._fh = file
            try:
                self._fh.tell()
            except Exception as exc:
                msg = f'{file=!r} is not seekable'
                raise TypeError(msg) from exc
            if hasattr(file, 'path'):
                self._path = os.path.normpath(file.path)
            elif hasattr(file, 'name'):
                self._path = os.path.normpath(file.name)

        elif hasattr(file, 'open'):
            # fsspec OpenFile
            self._fh = file.open()
            self._close = True
            try:
                self._fh.tell()
            except Exception as exc:
                with contextlib.suppress(Exception):
                    self._fh.close()
                msg = f'{file=!r} is not seekable'
                raise ValueError(msg) from exc
            if hasattr(file, 'path'):
                self._path = os.path.normpath(file.path)

        else:
            msg = f'cannot handle {type(file)=}'
            raise ValueError(msg)

        if hasattr(file, 'name') and file.name:
            self._name = os.path.basename(file.name)
        elif self._path:
            self._name = os.path.basename(self._path)
        elif isinstance(file, io.BytesIO):
            self._name = 'BytesIO'
        # else:
        #     self._name = f'{type(file)}'

    @property
    def filehandle(self) -> IO[bytes]:
        """File handle."""
        return self._fh

    @property
    def filepath(self) -> str:
        """Path to file or empty if binary stream."""
        return self._path

    @property
    def filename(self) -> str:
        """Name of file or empty if binary stream."""
        return os.path.basename(self._path)

    @property
    def dirname(self) -> str:
        """Directory containing file or empty if binary stream."""
        return os.path.dirname(self._path)

    @property
    def name(self) -> str:
        """Display name of file."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def attrs(self) -> dict[str, Any]:
        """Selected metadata as dict."""
        return {'name': self.name, 'filepath': self.filepath}

    @property
    def closed(self) -> bool:
        """File is closed."""
        return self._closed

    def close(self) -> None:
        """Close file."""
        if self._close:
            self._closed = True
            with contextlib.suppress(Exception):
                self._fh.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def __repr__(self) -> str:
        if self._name:
            return f'<{self.__class__.__name__} {self._name!r}>'
        return f'<{self.__class__.__name__}>'


@final
class ObfFile(BinaryFile):
    """Object Binary Format file.

    ``ObfFile`` instances are not thread-safe. All attributes are read-only.

    ``ObfFile`` instances must be closed with :py:meth:`ObfFile.close`,
    which is automatically called when using the 'with' context manager.

    Parameters:
        file:
            Name of Object Binary Format file or seekable binary stream.
        mode:
            File open mode if `file` is file name.
            The default is 'r'. Files are always opened in binary mode.
        squeeze:
            Remove dimensions of length one from stacks.

    Raises:
        ObfFileError: File is not in Object Binary Format or is corrupted.

    """

    header: ObfFileHeader
    """Parsed OBF file header."""

    _squeeze: bool  # remove dimensions of length one from stacks

    def __init__(
        self,
        file: str | os.PathLike[Any] | IO[bytes],
        /,
        *,
        squeeze: bool = True,
        mode: Literal['r', 'r+'] | None = None,
    ) -> None:
        super().__init__(file, mode=mode)

        self._squeeze = bool(squeeze)
        try:
            self.header = ObfFileHeader.fromfile(self._fh)
        except ObfFileError:
            self.close()
            raise
        except Exception as exc:
            self.close()
            raise ObfFileError('invalid Imspector image file') from exc

    @cached_property
    def stacks(self) -> ObfStackSequence:
        """Sequence of image stacks in file."""
        return ObfStackSequence(self)

    def __enter__(self) -> Self:
        return self

    def __str__(self) -> str:
        return indent(
            repr(self),
            f'path: {self._path}',
            self.header,
            self.stacks,
        )


class ObfStack:
    """Image stack inside OBF file."""

    header: ObfStackHeader
    """Parsed stack header."""

    footer: ObfStackFooter
    """Parsed stack footer."""

    def __init__(
        self,
        fh: IO[bytes],
        *,
        squeeze: bool,
    ) -> None:
        self._fh = fh
        self._squeeze = squeeze

        self.header = ObfStackHeader.fromfile(fh)
        self.footer = ObfStackFooter.fromfile(fh, self.header)

        # parse dtype and samples-per-pixel from header flags
        dtype_flags = self.header.data_type
        is_complex = bool(dtype_flags & 0x40000000)
        if is_complex:
            dtype_flags ^= 0x40000000

        self._spp = {
            0x00000400: 3,  # RGB
            0x00000800: 4,  # RGBA
        }.get(dtype_flags, 1)

        dtype = {
            0x00000001: numpy.uint8,
            0x00000002: numpy.int8,
            0x00000004: numpy.uint16,
            0x00000008: numpy.int16,
            0x00000010: numpy.uint32,
            0x00000020: numpy.int32,
            0x00000040: numpy.float32,
            0x00000080: numpy.float64,
            0x00000400: numpy.uint8,  # RGB
            0x00000800: numpy.uint8,  # RGBA
            0x00001000: numpy.uint64,
            0x00002000: numpy.int64,
            0x00010000: numpy.bool_,
        }.get(dtype_flags)
        if dtype is None:
            raise ObfFileError(f'unknown OMAS_DT flags {dtype_flags:#010x}')
        if is_complex:
            if dtype == numpy.float32:
                dtype = numpy.complex64
            elif dtype == numpy.float64:
                dtype = numpy.complex128
            else:
                raise ObfFileError(
                    'complex flag is only valid for float32 and float64'
                )
        self._dtype = numpy.dtype(dtype)

    @property
    def name(self) -> str:
        """Stack name."""
        return self.header.name

    @cached_property
    def dtype(self) -> numpy.dtype[Any]:
        """NumPy data type of image stack."""
        return self._dtype

    @cached_property
    def sizes(self) -> dict[str, int]:
        """Ordered mapping of dimension name to length."""
        # Dimension names are read from footer labels. If a label is empty,
        # it is replaced with a default name from 'XYZCT' or a numeric index.
        # TODO: get dimension order from OME metadata DimensionOrder?
        hdr = self.header
        raw_shape = list(hdr.res)  # len == rank, stored as [d0, d1, ..., dN]
        raw_labels = list(self.footer.labels)
        if len(raw_labels) < len(raw_shape):
            raw_labels += [
                str(i) for i in range(len(raw_labels), len(raw_shape))
            ]
        # OBF stores dimension 0 first (X is fastest on disk)
        shape_rev = raw_shape[::-1]
        labels_rev = raw_labels[::-1]
        orig_idx_rev = list(range(hdr.rank - 1, -1, -1))
        seen: set[str] = set()
        result: dict[str, int] = {}
        for orig_i, size, lbl in zip(
            orig_idx_rev, shape_rev, labels_rev, strict=True
        ):
            if self._squeeze and size == 1:
                continue
            key = lbl.strip()
            if not key:
                key = 'XYZCT'[orig_i] if orig_i < 5 else str(orig_i)
            base = key
            n = 2
            while key in seen:
                key = f'{base}{n}'
                n += 1
            seen.add(key)
            result[key] = size
        if self._spp > 1:
            result['S'] = self._spp
        return result

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of image stack."""
        return tuple(self.sizes.values())

    @property
    def dims(self) -> tuple[str, ...]:
        """Dimension names of image stack."""
        return tuple(self.sizes.keys())

    @property
    def ndim(self) -> int:
        """Number of image stack dimensions."""
        return len(self.sizes)

    @property
    def nbytes(self) -> int:
        """Number of bytes consumed by image stack."""
        size = 1
        for i in self.sizes.values():
            size *= i
        return size * self.dtype.itemsize

    @property
    def size(self) -> int:
        """Number of elements in image stack."""
        size = 1
        for i in self.sizes.values():
            size *= i
        return size

    @cached_property
    def coords(self) -> dict[str, NDArray[Any]]:
        """Mapping of dimension names to physical coordinate arrays."""
        hdr = self.header
        dim_keys = [k for k in self.sizes if k != 'S']
        raw_shape = list(hdr.res)
        orig_idx_rev = range(hdr.rank - 1, -1, -1)
        orig_indices = [
            orig_i
            for orig_i, size in zip(orig_idx_rev, raw_shape[::-1], strict=True)
            if not (self._squeeze and size == 1)
        ]
        # TODO: use footer si_dimension and si_dimensions?
        result: dict[str, NDArray[Any]] = {}
        for orig_i, key in zip(orig_indices, dim_keys, strict=True):
            size = hdr.res[orig_i]
            if orig_i in self.footer.col_positions:
                result[key] = numpy.asarray(self.footer.col_positions[orig_i])
            elif size > 0:
                length = (
                    hdr.lengths[orig_i] if orig_i < len(hdr.lengths) else 0.0
                )
                offset = (
                    hdr.offsets[orig_i] if orig_i < len(hdr.offsets) else 0.0
                )
                result[key] = numpy.linspace(
                    offset, offset + length, size, endpoint=False
                )
        return result

    @cached_property
    def attrs(self) -> dict[str, Any]:
        """Image stack metadata as dict."""
        attrs: dict[str, Any] = {
            'description': self.header.description,
        }
        if self.footer.si_value is not None:
            attrs['si_value'] = str(self.footer.si_value)
        if self.footer.si_dimensions:
            hdr = self.header
            si_dims = self.footer.si_dimensions[: hdr.rank][::-1]
            attrs['si_dimensions'] = tuple(
                str(s)
                for s, size in zip(si_dims, list(hdr.res)[::-1], strict=True)
                if not (self._squeeze and size == 1)
            )
        if self.footer.metadata:
            attrs['metadata'] = self.footer.metadata
        if self.footer.tag_dictionary:
            attrs['tag_dictionary'] = self.footer.tag_dictionary
        return attrs

    def asarray(self) -> NDArray[Any]:
        """Return image stack data as NumPy array."""
        hdr = self.header
        ftr = self.footer
        fh = self._fh

        num_chunks = ftr.num_chunk_positions or 0

        if num_chunks == 0:
            fh.seek(hdr.data_pos)
            raw = fh.read(hdr.data_length)
        else:
            # chunked / interleaved read (format version 6)
            #
            # chunk_positions[i] = (lo, fo) where:
            #   lo = cumulative compressed bytes read BEFORE seeking
            #   fo = file offset relative to data_pos to seek to
            #
            # The data is stored in non-contiguous file fragments.
            # The algorithm splices the fragments into one byte string
            # which is then decompressed once.
            end_disk = ftr.stack_end_used_disk or (
                hdr.data_pos + hdr.data_length
            )
            last_lo, last_fo = ftr.chunk_positions[-1]

            # total compressed bytes: bytes before last fragment + bytes in
            # last fragment up to stack_end_used_disk
            bytes_written = last_lo + (end_disk - hdr.data_pos - last_fo)

            pos = 0
            idx = 0
            seek_pos = hdr.data_pos
            fh.seek(seek_pos)
            pieces: list[bytes] = []
            while pos < bytes_written:
                n = bytes_written - pos
                if idx < len(ftr.chunk_positions):
                    lo, fo = ftr.chunk_positions[idx]
                    if pos + n > lo:
                        n = lo - pos
                        seek_pos = fo + hdr.data_pos
                        idx += 1
                if n > 0:
                    pieces.append(fh.read(n))
                fh.seek(seek_pos)
                pos += n
            raw = b''.join(pieces)

        if hdr.compression_type == 1:
            raw = zlib.decompressobj().decompress(raw)

        # trim trailing partial element (e.g. uncompressed chunked stacks
        # where bytes_written is not a multiple of the element size)
        itemsize = self._dtype.itemsize
        if len(raw) % itemsize:
            raw = raw[: len(raw) // itemsize * itemsize]

        arr = numpy.frombuffer(raw, dtype=self._dtype)

        # trim oversized array (shouldn't normally occur)
        total_expected = self.size
        if arr.size > total_expected:
            arr = arr[:total_expected]

        # pad undersized (partially written) array with zeros
        if arr.size < total_expected:
            arr = numpy.concatenate(
                [
                    arr,
                    numpy.zeros(total_expected - arr.size, dtype=self._dtype),
                ]
            )

        return arr.reshape(self.shape)

    def asxarray(self) -> DataArray:
        """Return image stack as xarray DataArray."""
        from xarray import DataArray

        return DataArray(
            self.asarray(),
            coords=self.coords,
            dims=self.dims,
            name=self.name,
            attrs=self.attrs,
        )

    def __str__(self) -> str:
        return indent(
            repr(self)[1:-1],
            self.header,
            self.footer,
        )

    def __repr__(self) -> str:
        dims = ', '.join(f'{k}: {v}' for k, v in self.sizes.items())
        return f'<ObfStack {self.name!r} ({dims}) {self._dtype}>'


@final
class ObfStackSequence(Sequence[ObfStack]):
    """Sequence of stacks in OBF file."""

    __slots__ = ('_parent', '_stacks')

    _parent: ObfFile
    _stacks: dict[str, ObfStack]

    def __init__(self, parent: ObfFile, /) -> None:
        self._parent = parent
        self._stacks = {}

        next_pos = parent.header.first_stack_pos
        while next_pos != 0:
            parent._fh.seek(next_pos)
            stack = ObfStack(parent._fh, squeeze=parent._squeeze)
            self._stacks[stack.name] = stack
            next_pos = stack.header.next_stack_pos

    def find(
        self,
        key: str,
        /,
        *,
        flags: int = re.IGNORECASE,
        default: Any = None,
    ) -> ObfStack | None:
        """Return first stack with name matching pattern, if any.

        Parameters:
            key:
                Regular expression pattern to match ObfStack name.
            flags:
                Regular expression flags.
            default:
                Value to return if no stack with matching name found.

        """
        pattern = re.compile(key, flags=flags)
        for stack in self._stacks.values():
            value = stack.name
            if pattern.search(value) is not None:
                return stack
        return default  # type: ignore[no-any-return]

    def findall(
        self,
        key: str,
        /,
        *,
        flags: int = re.IGNORECASE,
    ) -> tuple[ObfStack, ...]:
        """Return all stacks with name matching pattern.

        Parameters:
            key:
                Regular expression pattern to match ObfStack name.
            flags:
                Regular expression flags.

        """
        pattern = re.compile(key, flags=flags)
        images = []
        for stack in self._stacks.values():
            value = stack.name
            if pattern.search(value) is not None:
                images.append(stack)
        return tuple(images)

    def __getitem__(  # type: ignore[override]
        self,
        key: int | str,
        /,
    ) -> ObfStack:
        """Return stack at index or first stack with name matching pattern.

        Raises:
            IndexError: if integer index out of range.
            KeyError: if no stack with matching name pattern found.

        """
        if isinstance(key, int):
            index = key
            try:
                key = tuple(self._stacks.keys())[index]
            except IndexError:
                msg = f'stack {index=} out of range'
                raise IndexError(msg) from None
            return self._stacks[key]
        if key in self._stacks:
            return self._stacks[key]
        pattern = re.compile(key, flags=re.IGNORECASE)
        for stack in self._stacks.values():
            if pattern.search(stack.name) is not None:
                return stack
        msg = f'stack {key!r} not found'
        raise KeyError(msg)

    def __len__(self) -> int:
        return len(self._stacks)

    def __iter__(self) -> Iterator[ObfStack]:
        return iter(self._stacks.values())

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__} len={len(self._stacks)}>'

    def __str__(self) -> str:
        return indent(repr(self)[1:-1], *(self._stacks.values()))


@dataclass(slots=True, frozen=True)
class ObfFileHeader:
    """OBF file header (``OMAS_BF``)."""

    _FMT: ClassVar[struct.Struct] = struct.Struct('<10sIQI')

    format_version: int
    """File format version."""

    description: str
    """File-level XML description string."""

    first_stack_pos: int
    """File offset of first stack header."""

    meta_data_position: int | None
    """File offset of file-level tag dictionary, or ``None``."""

    metadata: dict[str, str]
    """File-level tag dictionary (for example, ``'ome_xml'``)."""

    def __str__(self) -> str:
        return dataclass_str(self)

    @classmethod
    def fromfile(cls, fh: IO[bytes]) -> Self:
        """Read and validate OBF file header from open file."""
        raw = fh.read(cls._FMT.size)
        magic, fmt_ver, first_stack_pos, descr_len = cls._FMT.unpack(raw)
        if magic != b'OMAS_BF\n\xff\xff':
            raise ObfFileError(f'invalid file magic: {magic!r}')
        description = read_strn(fh, descr_len)
        meta_data_position: int | None = None
        metadata: dict[str, str] = {}
        if fmt_ver >= 2:
            meta_data_position = int(struct.unpack('<Q', fh.read(8))[0])
            pos_restore = fh.tell()
            fh.seek(meta_data_position)
            metadata = read_tag_dict(fh)
            fh.seek(pos_restore)
        return cls(
            format_version=int(fmt_ver),
            description=description,
            first_stack_pos=int(first_stack_pos),
            meta_data_position=meta_data_position,
            metadata=metadata,
        )


@dataclass(slots=True, frozen=True)
class ObfStackHeader:
    """OBF stack header (``OMAS_BF_STACK``)."""

    _FMT: ClassVar[struct.Struct] = struct.Struct('<16s17I30d5I3Q')

    format_version: int
    """Stack format version."""

    rank: int
    """Number of active dimensions."""

    res: tuple[int, ...]
    """Number of pixels along each dimension (first ``rank`` values)."""

    lengths: tuple[float, ...]
    """Physical length along each dimension in metres."""

    offsets: tuple[float, ...]
    """Physical offset along each dimension in metres."""

    data_type: int
    """OMAS_DT flags encoding element type."""

    compression_type: int
    """0 = uncompressed, 1 = zlib."""

    compression_level: int
    """Zlib compression level."""

    name: str
    """Stack name."""

    description: str
    """Stack XML description string."""

    data_pos: int
    """File offset of first data byte."""

    data_length: int
    """Number of data bytes on disk."""

    next_stack_pos: int
    """File offset of next stack header, or 0 if last stack."""

    def __str__(self) -> str:
        return dataclass_str(self)

    @classmethod
    def fromfile(cls, fh: IO[bytes]) -> Self:
        """Read OBF stack header from open file."""
        position = fh.tell()
        vals = cls._FMT.unpack(fh.read(cls._FMT.size))
        magic = vals[0]
        if magic != b'OMAS_BF_STACK\n\xff\xff':
            raise ObfFileError(
                f'invalid stack magic at offset {position}: {magic!r}'
            )
        fmt_ver = int(vals[1])
        rank = int(vals[2])
        res = tuple(int(v) for v in vals[3:18])
        lengths = tuple(float(v) for v in vals[18:33])
        offsets = tuple(float(v) for v in vals[33:48])
        data_type = int(vals[48])
        compression_type = int(vals[49])
        compression_level = int(vals[50])
        name_len = int(vals[51])
        descr_len = int(vals[52])
        # vals[53] = reserved (uint64)
        data_length = int(vals[54])
        next_stack_pos = int(vals[55])
        name = read_strn(fh, name_len)
        description = read_strn(fh, descr_len)
        data_pos = fh.tell()
        return cls(
            format_version=fmt_ver,
            rank=rank,
            res=res[:rank],
            lengths=lengths[:rank],
            offsets=offsets[:rank],
            data_type=data_type,
            compression_type=compression_type,
            compression_level=compression_level,
            name=name,
            description=description,
            data_pos=data_pos,
            data_length=data_length,
            next_stack_pos=next_stack_pos,
        )


@dataclass(slots=True, frozen=False)
class ObfStackFooter:
    """OBF stack footer (version-incremental)."""

    # V1
    size: int = 0
    """Total size of fixed footer region in bytes."""

    has_col_positions: tuple[int, ...] = field(default_factory=tuple)
    """Per-axis flag: non-zero if physical column positions are stored."""

    has_col_labels: tuple[int, ...] = field(default_factory=tuple)
    """Per-axis flag: non-zero if column label strings are stored."""

    metadata_length: int = 0
    """Length in bytes of legacy metadata string."""

    # V2
    si_value: ObfSiUnit | None = None
    """SI unit of pixel (sample) values."""

    si_dimensions: list[ObfSiUnit] = field(default_factory=list)
    """SI units of coordinate axes."""

    # V3
    num_flush_points: int = 0
    """Number of zlib flush points."""

    flush_block_size: int | None = None
    """Block size between flush points."""

    # V4
    tag_dictionary_length: int | None = None
    """Byte length of stack-level tag dictionary."""

    # V5
    stack_end_disk: int | None = None
    """File offset past end of complete stack."""

    min_format_version: int | None = None
    """Minimum format version required to read this stack."""

    # V5a
    stack_end_used_disk: int | None = None
    """File offset past actually written data."""

    # V6
    samples_written: int | None = None
    """Number of samples actually written (0 = all expected)."""

    num_chunk_positions: int | None = None
    """Number of interleaved-chunk position entries."""

    # post-footer variable-length data
    labels: list[str] = field(default_factory=list)
    """Axis label strings for each dimension."""

    col_positions: dict[int, tuple[float, ...]] = field(default_factory=dict)
    """Physical position arrays keyed by axis index."""

    col_labels: dict[int, list[str]] = field(default_factory=dict)
    """Column label strings keyed by axis index."""

    metadata: str = ''
    """Legacy metadata string."""

    flush_positions: list[int] = field(default_factory=list)
    """File offsets of zlib flush points."""

    tag_dictionary: dict[str, str] = field(default_factory=dict)
    """Stack-level tag dictionary."""

    chunk_positions: list[tuple[int, int]] = field(default_factory=list)
    """Interleaved-chunk entries as (logical_offset, file_offset) pairs."""

    def __str__(self) -> str:
        return dataclass_str(self)

    @classmethod
    def fromfile(cls, fh: IO[bytes], header: ObfStackHeader) -> Self:
        """Read OBF stack footer from open file."""
        fmt_ver = header.format_version
        rank = header.rank
        footer_pos = header.data_pos + header.data_length
        self = cls()

        if fmt_ver >= 1:
            fh.seek(footer_pos)
            # V1 fixed block:
            # size(1) + has_col_positions(15) + has_col_labels(15)
            #   + metadata_length(1) = 32 uint32s
            v1 = struct.unpack('<32I', fh.read(128))
            self.size = int(v1[0])
            self.has_col_positions = tuple(int(x) for x in v1[1:16])[:rank]
            self.has_col_labels = tuple(int(x) for x in v1[16:31])[:rank]
            self.metadata_length = int(v1[31])

            if fmt_ver >= 2:
                # V2: si_value + si_dimensions[15], each <18id> (80 bytes)
                self.si_value = ObfSiUnit.fromfile(fh)
                # OMAS_BF_MAX_DIMENSIONS = 15
                self.si_dimensions = [
                    ObfSiUnit.fromfile(fh) for _ in range(15)
                ]

            if fmt_ver >= 3:
                # V3: num_flush_points, flush_block_size
                self.num_flush_points, self.flush_block_size = struct.unpack(
                    '<2Q', fh.read(16)
                )

            if fmt_ver >= 4:
                # V4: tag_dictionary_length
                self.tag_dictionary_length = struct.unpack('<Q', fh.read(8))[0]

            if fmt_ver >= 5:
                # V5: stack_end_disk (uint64) + min_format_version (uint32)
                self.stack_end_disk, self.min_format_version = struct.unpack(
                    '<QI', fh.read(12)
                )

            if fmt_ver >= 6:
                # V5a: stack_end_used_disk
                self.stack_end_used_disk = struct.unpack('<Q', fh.read(8))[0]
                # V6: samples_written, num_chunk_positions
                self.samples_written, self.num_chunk_positions = struct.unpack(
                    '<2Q', fh.read(16)
                )

            # seek past fixed footer region (skip unknown future fields)
            fh.seek(footer_pos + self.size)

            # post-footer variable-length data:

            # 1. axis labels (rank x length-prefixed string)
            self.labels = [read_strn(fh) for _ in range(rank)]

            # 2. column positions (doubles) for flagged axes
            for i, has_pos in enumerate(self.has_col_positions):
                if has_pos:
                    n = header.res[i]
                    self.col_positions[i] = struct.unpack(
                        f'<{n}d', fh.read(n * 8)
                    )

            # 3. column labels (strings) for flagged axes
            for i, has_lbl in enumerate(self.has_col_labels):
                if has_lbl:
                    n = header.res[i]
                    self.col_labels[i] = [read_strn(fh) for _ in range(n)]

            # 4. legacy metadata string
            if self.metadata_length > 0:
                self.metadata = read_strn(fh, self.metadata_length)

            # 5. flush positions
            for _ in range(self.num_flush_points):
                fp = struct.unpack('<Q', fh.read(8))[0]
                self.flush_positions.append(fp)

            # 6. tag dictionary
            if self.tag_dictionary_length:
                buf = io.BytesIO(fh.read(int(self.tag_dictionary_length)))
                self.tag_dictionary = read_tag_dict(buf)

            # 7. chunk positions: (logical_offset, file_offset) pairs
            num_chunks = self.num_chunk_positions or 0
            for _ in range(num_chunks):
                lo, fo = struct.unpack('<2Q', fh.read(16))
                self.chunk_positions.append((lo, fo))

        return self


@dataclass(slots=True, frozen=True)
class ObfSiUnit:
    """SI unit stored as rational exponents and scale factor.

    Each base-unit field is a ``(numerator, denominator)`` int32 pair.
    """

    _FMT: ClassVar[struct.Struct] = struct.Struct('<18id')

    meters: tuple[int, int]
    kilograms: tuple[int, int]
    seconds: tuple[int, int]
    amperes: tuple[int, int]
    kelvin: tuple[int, int]
    moles: tuple[int, int]
    candela: tuple[int, int]
    radian: tuple[int, int]
    steradian: tuple[int, int]
    scale_factor: float

    def __str__(self) -> str:
        names = ('m', 'kg', 's', 'A', 'K', 'mol', 'cd', 'rad', 'sr')
        values = (
            self.meters,
            self.kilograms,
            self.seconds,
            self.amperes,
            self.kelvin,
            self.moles,
            self.candela,
            self.radian,
            self.steradian,
        )
        parts = []
        for (n, d), name in zip(values, names, strict=True):
            if n == 0:
                continue
            exp = f'{n}/{d}' if d not in (0, 1) else str(n)
            parts.append(name if exp == '1' else f'{name}^{exp}')
        sf = self.scale_factor
        if not parts:
            return '' if sf == 1.0 else f'{sf:g}'
        if sf == 1.0:
            return ' '.join(parts)
        return f'{sf:g} {" ".join(parts)}'

    @classmethod
    def fromfile(cls, fh: IO[bytes]) -> ObfSiUnit:
        """Read one SI unit record from *fh*."""
        values = cls._FMT.unpack(fh.read(cls._FMT.size))
        fracs = [
            (int(values[i * 2]), int(values[i * 2 + 1])) for i in range(9)
        ]
        return cls(
            meters=fracs[0],
            kilograms=fracs[1],
            seconds=fracs[2],
            amperes=fracs[3],
            kelvin=fracs[4],
            moles=fracs[5],
            candela=fracs[6],
            radian=fracs[7],
            steradian=fracs[8],
            scale_factor=float(values[18]),
        )


FILE_EXTENSIONS = {
    '.obf': 'OBF files',
    '.msr': 'MSR files',
}
"""Supported file extensions of Imspector image files."""


def read_strn(fh: IO[bytes], length: int | None = None) -> str:
    """Read fixed-length or length-prefixed UTF-8 string from file."""
    if length is None:
        length = struct.unpack('<I', fh.read(4))[0]
    data = fh.read(length)
    try:
        return data.decode('utf-8')
    except UnicodeDecodeError:
        return data.decode('iso-8859-1')


def read_tag_dict(fh: IO[bytes]) -> dict[str, str]:
    """Read null-terminated key/value UTF-8 string pairs from file."""
    tag_dict: dict[str, str] = {}
    while True:
        n = struct.unpack('<I', fh.read(4))[0]
        if n == 0:
            break
        key = read_strn(fh, n)
        tag_dict[key] = read_strn(fh)
    return tag_dict


def indent(*args: Any) -> str:
    """Return joined string representations of objects with indented lines."""
    text = '\n'.join(str(arg) for arg in args)
    return '\n'.join(
        ('  ' + line if line else line) for line in text.splitlines() if line
    )[2:]


def dataclass_str(obj: Any, /) -> str:
    """Return string representation of dataclass, skipping default fields."""

    def fmt_item(val: Any, /) -> str:
        return repr(str(val)) if isinstance(val, ObfSiUnit) else repr(val)

    lines = [type(obj).__name__]
    for f in dataclasses.fields(obj):
        value = getattr(obj, f.name)
        if f.default is not dataclasses.MISSING and value == f.default:
            continue
        if (
            f.default_factory is not dataclasses.MISSING
            and value == f.default_factory()
        ):
            continue
        if isinstance(value, (list, tuple)):
            items = ', '.join(fmt_item(x) for x in value)
            brackets = '()' if isinstance(value, tuple) else '[]'
            val = f'{brackets[0]}{items}{brackets[1]}'
        else:
            val = fmt_item(value)
        lines.append(
            snipstr(f'{f.name}: {val}', 74, ellipsis='...', snipat=1.0)
        )
    return indent(*lines)


def snipstr(
    string: str,
    /,
    width: int = 79,
    *,
    snipat: float | None = None,
    ellipsis: str | None = None,
) -> str:
    """Return string cut to specified length.

    Parameters:
        string:
            String to snip.
        width:
            Maximum length of returned string.
        snipat:
            Approximate position at which to split long strings.
            The default is 0.5.
        ellipsis:
            Characters to insert between splits of long strings.
            The default is '\u2026'.

    Examples:
        >>> snipstr('abcdefghijklmnop', 8, ellipsis='...')
        'abc...op'

    """
    if snipat is None:
        snipat = 0.5
    if ellipsis is None:
        ellipsis = '\u2026'
    esize = len(ellipsis)

    splitlines = string.splitlines()
    # TODO: finish and test multiline snip

    result = []
    for line in splitlines:
        linelen = len(line)
        if linelen <= width:
            result.append(line)
            continue

        if snipat == 1:
            split = linelen
        elif 0 < abs(snipat) < 1:
            split = math.floor(linelen * snipat)
        else:
            split = int(snipat)

        if split < 0:
            split += linelen
            split = max(split, 0)

        if esize == 0 or width < esize + 1:
            if split <= 0:
                result.append(line[-width:])
            else:
                result.append(line[:width])
        elif split <= 0:
            result.append(ellipsis + line[esize - width :])
        elif split >= linelen or width < esize + 4:
            result.append(line[: width - esize] + ellipsis)
        else:
            splitlen = linelen - width + esize
            end1 = split - splitlen // 2
            end2 = end1 + splitlen
            result.append(line[:end1] + ellipsis + line[end2:])

    return '\n'.join(result)


def askopenfilename(**kwargs: Any) -> str:
    """Return file name(s) from Tkinter's file open dialog."""
    from tkinter import Tk, filedialog

    root = Tk()
    root.withdraw()
    root.update()
    filenames = filedialog.askopenfilename(**kwargs)
    root.destroy()
    return filenames


def main(argv: list[str] | None = None) -> int:
    """Command line usage main function.

    Preview image and metadata in specified files or all files in directory.

    ``python -m obffile file_or_directory``

    """
    from glob import glob

    imshow: Any
    try:
        from tifffile import imshow
    except ImportError:
        imshow = None

    xarray: Any
    try:
        import xarray
    except ImportError:
        xarray = None

    if argv is None:
        argv = sys.argv

    if len(argv) == 1:
        path = askopenfilename(
            title='Select an Imspector image file',
            filetypes=[
                (f'{desc}', f'*{ext}') for ext, desc in FILE_EXTENSIONS.items()
            ]
            + [('All files', '*')],
        )
        files = [path] if path else []
    elif '*' in argv[1]:
        files = glob(argv[1])
    elif os.path.isdir(argv[1]):
        files = [
            f
            for ext in FILE_EXTENSIONS
            for f in glob(f'{argv[1]}/**/*{ext}', recursive=True)
        ]
    else:
        files = argv[1:]

    for fname in files:
        try:
            with ObfFile(fname) as obf:
                print(obf)
                print()
                if imshow is None:
                    continue
                for i, stack in enumerate(obf.stacks):
                    im: Any
                    try:
                        if xarray is not None:
                            im = stack.asxarray()
                            data = im.data
                        else:
                            im = stack.asarray()
                            data = im
                        print(im)
                        print()
                        if data.ndim < 2:
                            continue
                        pm = (
                            'RGB'
                            if stack.dims[-1:] == ('S',)
                            else 'MINISBLACK'
                        )
                        imshow(
                            data,
                            title=repr(stack),
                            show=i == len(obf.stacks) - 1,
                            photometric=pm,
                            interpolation='None',
                        )
                    except Exception as exc:
                        print(fname, exc)
        except Exception:
            import traceback

            print('Failed to read', fname)
            traceback.print_exc()
            print()
            continue

    return 0


if __name__ == '__main__':
    sys.exit(main())
