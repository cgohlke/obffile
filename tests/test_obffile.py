# test_obffile.py

# Copyright (c) 2025-2026, Christoph Gohlke
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the copyright holders nor the names of any
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Unittests for the obffile package.

:Version: 2026.6.28

"""

import contextlib
import glob
import io
import itertools
import os
import pathlib
import sys
import sysconfig

import numpy
import pytest
from numpy.testing import assert_allclose
from xarray import DataArray

try:
    import fsspec
except ImportError:
    fsspec = None  # type: ignore[assignment]

import obffile
from obffile import (
    FILE_EXTENSIONS,
    ObfFile,
    ObfFileError,
    ObfFileHeader,
    ObfSiUnit,
    ObfStack,
    ObfStackFooter,
    ObfStackHeader,
    ObfStackSequence,
    __version__,
    imread,
)
from obffile.obffile import BinaryFile

HERE = pathlib.Path(os.path.dirname(__file__))
DATA = HERE / 'data'


@pytest.mark.skipif(__doc__ is None, reason='__doc__ is None')
def test_version():
    """Assert obffile versions match docstrings."""
    ver = ':Version: ' + __version__
    assert ver in __doc__
    assert ver in obffile.__doc__


class TestBinaryFile:
    """Test BinaryFile with different file-like inputs."""

    def setup_method(self):
        self.filename = os.path.normpath(DATA / 'binary.bin')
        if not os.path.exists(self.filename):
            pytest.skip(f'{self.filename!r} not found')

    def validate(
        self,
        fh: BinaryFile,
        filepath: str | None = None,
        filename: str | None = None,
        dirname: str | None = None,
        name: str | None = None,
        *,
        closed: bool = True,
    ) -> None:
        """Assert BinaryFile attributes."""
        if filepath is None:
            filepath = self.filename
        if filename is None:
            filename = os.path.basename(self.filename)
        if dirname is None:
            dirname = os.path.dirname(self.filename)
        if name is None:
            name = fh.filename

        attrs = fh.attrs
        assert attrs['name'] == name
        assert attrs['filepath'] == filepath

        assert fh.filepath == filepath
        assert fh.filename == filename
        assert fh.dirname == dirname
        assert fh.name == name
        assert fh.closed is False
        assert fh.filesize == 256
        assert len(fh.filehandle.read()) == 256
        fh.filehandle.seek(10)
        assert fh.filehandle.tell() == 10
        assert fh.filehandle.read(1) == b'\n'
        fh.close()
        # underlying filehandle may still be be open if
        # BinaryFile was given an open filehandle
        assert fh._fh.closed is closed
        # BinaryFile always reports itself as closed after close() is called
        assert fh.closed

    def test_str(self):
        """Test BinaryFile with str path."""
        file = self.filename
        with BinaryFile(file) as fh:
            self.validate(fh, closed=True)

    def test_pathlib(self):
        """Test BinaryFile with pathlib.Path."""
        file = pathlib.Path(self.filename)
        with BinaryFile(file) as fh:
            self.validate(fh, closed=True)

    def test_open_file(self):
        """Test BinaryFile with open binary file."""
        with open(self.filename, 'rb') as fh, BinaryFile(fh) as bf:
            self.validate(bf, closed=False)

    def test_bytesio(self):
        """Test BinaryFile with BytesIO."""
        with open(self.filename, 'rb') as fh:
            file = io.BytesIO(fh.read())
        with BinaryFile(file) as fh:
            self.validate(
                fh,
                filepath='',
                filename='',
                dirname='',
                name='BytesIO',
                closed=False,
            )

    @pytest.mark.skipif(fsspec is None, reason='fsspec not installed')
    def test_fsspec_openfile(self):
        """Test BinaryFile with fsspec OpenFile."""
        file = fsspec.open(self.filename)
        with BinaryFile(file) as fh:
            self.validate(fh, closed=True)

    @pytest.mark.skipif(fsspec is None, reason='fsspec not installed')
    def test_fsspec_localfileopener(self):
        """Test BinaryFile with fsspec LocalFileOpener."""
        with fsspec.open(self.filename) as file, BinaryFile(file) as fh:
            self.validate(fh, closed=False)

    def test_text_file_fails(self):
        """Test BinaryFile with open text file fails."""
        with open(self.filename) as fh:  # noqa: SIM117
            with pytest.raises(TypeError):
                BinaryFile(fh)

    def test_file_extension_fails(self):
        """Test BinaryFile with wrong file extension fails."""
        ext = BinaryFile._ext
        BinaryFile._ext = {'.lif'}
        try:
            with pytest.raises(ValueError):
                BinaryFile(self.filename)
        finally:
            BinaryFile._ext = ext

    def test_file_not_seekable(self):
        """Test BinaryFile with non-seekable file fails."""

        class File:
            # mock file object without tell methods
            def seek(self):
                pass

        with pytest.raises(ValueError):
            BinaryFile(File())

    def test_openfile_not_seekable(self):
        """Test BinaryFile with non-seekable file fails."""

        class File:
            # mock fsspec OpenFile without seek/tell methods
            @staticmethod
            def open(*args, **kwargs):
                del args, kwargs
                return File()

        with pytest.raises(ValueError):
            BinaryFile(File())

    def test_invalid_object(self):
        """Test BinaryFile with invalid file object fails."""

        class File:
            # mock non-file object
            pass

        with pytest.raises(TypeError):
            BinaryFile(File())

    def test_invalid_mode(self):
        """Test BinaryFile with invalid mode fails."""
        with pytest.raises(ValueError):
            BinaryFile(self.filename, mode='ab')

    def test_memmap_disabled(self):
        """Test memmap=False (default) does not memory-map file."""
        with BinaryFile(self.filename) as fh:
            assert fh._mm is None
        with BinaryFile(self.filename, memmap=False) as fh:
            assert fh._mm is None

    def test_memmap_bytesio_ignored(self):
        """Test memmap=True is silently ignored for BytesIO."""
        with open(self.filename, 'rb') as f:
            data = f.read()
        with BinaryFile(io.BytesIO(data), memmap=True) as fh:
            assert fh._mm is None
            assert bytes(fh._read_at(0, 4)) == b'\x00\x01\x02\x03'

    @pytest.mark.parametrize('memmap', [False, True])
    def test_read_at(self, memmap):
        """Test _read_at returns bytes or memoryview at offset."""
        with BinaryFile(self.filename, memmap=memmap) as fh:
            data = fh._read_at(0, 4)
            assert isinstance(data, memoryview if memmap else bytes)
            assert bytes(data) == b'\x00\x01\x02\x03'
            data = fh._read_at(10, 3)
            assert bytes(data) == b'\x0a\x0b\x0c'

    @pytest.mark.parametrize('memmap', [False, True])
    def test_read_array(self, memmap):
        """Test _read_array returns array at offset; dtype, count=-1, copy."""
        dtype = numpy.dtype('uint8')
        with BinaryFile(self.filename, memmap=memmap) as fh:
            arr = fh._read_array(0, 4, dtype)
            assert arr.dtype == dtype
            assert numpy.array_equal(arr, [0, 1, 2, 3])
            arr = fh._read_array(10, 3, dtype)
            assert numpy.array_equal(arr, [10, 11, 12])
            if memmap:
                assert not arr.flags.writeable
                arr_copy = fh._read_array(0, 4, dtype, copy=True)
                assert arr_copy.flags.writeable
                arr_copy[0] = 99  # must not raise
            else:
                # multi-byte dtype: binary.bin bytes as uint16 LE pairs
                arr16 = fh._read_array(0, 4, numpy.dtype('<u2'))
                assert numpy.array_equal(
                    arr16, [0x0100, 0x0302, 0x0504, 0x0706]
                )
            # count=-1 reads to end of file
            arr_end = fh._read_array(252, -1, dtype)
            assert numpy.array_equal(arr_end, [252, 253, 254, 255])

    def test_read_array_writable_memmap(self, tmp_path):
        """Test  writable=False makes mmap RO; writable=True keeps it RW."""
        tmpfile = tmp_path / 'test.bin'
        with open(self.filename, 'rb') as src:
            tmpfile.write_bytes(src.read())
        dtype = numpy.dtype('uint8')
        with BinaryFile(tmpfile, mode='r+', memmap=True) as fh:
            arr_ro = fh._read_array(0, 4, dtype)
            assert not arr_ro.flags.writeable
            arr_rw = fh._read_array(0, 4, dtype, writable=True)
            assert arr_rw.flags.writeable

    def test_read_array_truncate(self, caplog):
        """Test truncate=False raises; True returns partial; None logs."""
        import logging

        with (
            BinaryFile(io.BytesIO(bytes(range(5)))) as fh,
            pytest.raises(ValueError, match='expected 10 items, got 5'),
        ):
            fh._read_array(0, 10, numpy.uint8, truncate=False)
        with BinaryFile(io.BytesIO(bytes(range(5)))) as fh:
            arr = fh._read_array(0, 10, numpy.uint8, truncate=True)
            assert len(arr) == 5
            assert numpy.array_equal(arr, [0, 1, 2, 3, 4])
        logger = BinaryFile.__module__.split('.')[0]
        with (
            BinaryFile(io.BytesIO(bytes(range(5)))) as fh,
            caplog.at_level(logging.ERROR, logger=logger),
        ):
            arr = fh._read_array(0, 10, numpy.uint8, truncate=None)
        assert len(arr) == 5
        assert 'expected 10 items, got 5' in caplog.text

    def test_read_array_mmap_alignment(self, tmp_path):
        """Test mmap path truncates partial element to full boundary."""
        # 7 bytes; requesting 4 uint16 (8 bytes); 3 complete elements fit
        tmpfile = tmp_path / 'align.bin'
        tmpfile.write_bytes(b'\x01\x00\x02\x00\x03\x00\xff')
        dtype = numpy.dtype('<u2')
        with BinaryFile(tmpfile, memmap=True) as fh:
            arr = fh._read_array(0, 4, dtype, truncate=True)
            assert len(arr) == 3
            assert numpy.array_equal(arr, [1, 2, 3])
            # truncate=False must raise even on mmap path
            with pytest.raises(ValueError, match='expected 4 items, got 3'):
                fh._read_array(0, 4, dtype, truncate=False)

    def test_read_array_invalid_offset(self):
        """Test _read_array raises ValueError for negative offset."""
        with BinaryFile(io.BytesIO(bytes(range(8)))) as fh:  # noqa: SIM117
            with pytest.raises(ValueError, match='offset'):
                fh._read_array(-1, 4, numpy.uint8)

    def test_read_array_invalid_count(self):
        """Test _read_array raises ValueError for count < -1."""
        with BinaryFile(io.BytesIO(bytes(range(8)))) as fh:  # noqa: SIM117
            with pytest.raises(ValueError, match='count'):
                fh._read_array(0, -2, numpy.uint8)

    def test_write_at(self):
        """Test _write_at writes bytes to file at given offset."""
        with open(self.filename, 'rb') as fh:
            data = fh.read()
        file = io.BytesIO(data)
        with BinaryFile(file, mode='r+') as fh:
            fh._write_at(5, b'\xaa\xbb\xcc')
            assert bytes(fh._read_at(4, 5)) == b'\x04\xaa\xbb\xcc\x08'
        # verify persistence after close
        assert file.getvalue()[5:8] == b'\xaa\xbb\xcc'

    def test_write_at_memmap(self, tmp_path):
        """Test _write_at writes via mmap; mode='r+' creates writable mmap."""
        tmpfile = tmp_path / 'test.bin'
        tmpfile.write_bytes(pathlib.Path(self.filename).read_bytes())
        with BinaryFile(tmpfile, mode='r+', memmap=True) as fh:
            assert fh._mm is not None
            assert fh.writable
            assert bytes(fh._read_at(0, 4)) == b'\x00\x01\x02\x03'
            fh._write_at(5, b'\xaa\xbb\xcc')
        with BinaryFile(tmpfile) as fh:
            assert bytes(fh._read_at(5, 3)) == b'\xaa\xbb\xcc'

    def test_memmapped(self):
        """Test memmapped property reflects memory-map state."""
        with BinaryFile(self.filename) as fh:
            assert not fh.memmapped
        with BinaryFile(self.filename, memmap=True) as fh:
            assert fh.memmapped

    def test_writable(self, tmp_path):
        """Test writable property reflects file open mode."""
        tmpfile = tmp_path / 'test.bin'
        tmpfile.write_bytes(b'\x00' * 4)
        with BinaryFile(tmpfile) as fh:
            assert not fh.writable
        with BinaryFile(tmpfile, mode='r+') as fh:
            assert fh.writable

    def test_name_setter(self):
        """Test name setter updates display name and attrs."""
        with BinaryFile(self.filename) as fh:
            fh.name = 'custom'
            assert fh.name == 'custom'
            assert fh.attrs['name'] == 'custom'

    def test_set_lock(self):
        """Test set_lock enables and disables RLock; no-op when mmap active."""
        import threading

        with BinaryFile(self.filename) as fh:
            assert isinstance(fh._lock, contextlib.nullcontext)
            fh.set_lock(True)
            assert isinstance(fh._lock, type(threading.RLock()))
            fh.set_lock(False)
            assert isinstance(fh._lock, contextlib.nullcontext)
        with BinaryFile(self.filename, memmap=True) as fh:
            lock_before = fh._lock
            fh.set_lock(True)
            assert fh._lock is lock_before

    def test_repr(self):
        """Test __repr__ includes class name and file name."""
        with BinaryFile(self.filename) as fh:
            r = repr(fh)
            assert r.startswith('<BinaryFile ')
            assert 'binary.bin' in r


class TestObfFile:
    """Test ObfFile with different file-like inputs."""

    def setup_method(self):
        self.fname = os.path.normpath(DATA / 'Test.obf')
        if not os.path.exists(self.fname):
            pytest.skip(f'{self.fname!r} not found')

    def validate(self, obf: ObfFile, name: str = 'Test.obf') -> None:
        """Assert ObfFile attributes."""
        assert not obf.filehandle.closed
        assert obf.name == name
        assert obf.header.format_version == 2
        assert repr(obf).startswith('<ObfFile ')
        assert len(obf.stacks) == 7

    def test_str(self):
        """Test ObfFile with str path."""
        file = self.fname
        with ObfFile(file) as obf:
            self.validate(obf)

    def test_pathlib(self):
        """Test ObfFile with pathlib.Path."""
        file = pathlib.Path(self.fname)
        with ObfFile(file) as obf:
            self.validate(obf)

    def test_open_file(self):
        """Test ObfFile with open binary file."""
        with open(self.fname, 'rb') as fh, ObfFile(fh) as obf:
            self.validate(obf)

    def test_bytesio(self):
        """Test ObfFile with BytesIO."""
        with open(self.fname, 'rb') as fh:
            file = io.BytesIO(fh.read())
        with ObfFile(file) as obf:
            self.validate(obf, name='BytesIO')

    @pytest.mark.skipif(fsspec is None, reason='fsspec not installed')
    def test_fsspec_openfile(self):
        """Test ObfFile with fsspec OpenFile."""
        file = fsspec.open(self.fname)
        with ObfFile(file) as obf:
            self.validate(obf)

    @pytest.mark.skipif(fsspec is None, reason='fsspec not installed')
    def test_fsspec_localfileopener(self):
        """Test ObfFile with fsspec LocalFileOpener."""
        with fsspec.open(self.fname) as file, ObfFile(file) as obf:
            self.validate(obf)


def test_not_obf():
    """Test open non-OBF file raises exceptions."""
    with pytest.raises(ObfFileError):
        imread(DATA / 'empty.bin')
    with pytest.raises(TypeError):
        imread(TypeError)


@pytest.mark.parametrize('asxarray', [False, True])
def test_imread(asxarray):
    """Test imread function."""
    filename = DATA / 'Test.obf'

    data = imread(filename, 5, asxarray=asxarray)
    if asxarray:
        assert isinstance(data, DataArray)
        assert data.sizes == {'T': 18, 'Z': 2, 'Y': 339, 'X': 381}
        data = data.data
    else:
        assert isinstance(data, numpy.ndarray)
    assert data.shape == (18, 2, 339, 381)
    assert data.sum(dtype=numpy.uint32) == 599991


@pytest.mark.parametrize('memmap', [False, True])
@pytest.mark.parametrize('filetype', [str, io.BytesIO])
def test_obf(memmap, filetype):
    """Test OBF file."""
    filename = DATA / 'Test.obf'
    file = (
        filename if filetype is str else open(filename, 'rb')  # noqa: SIM115
    )

    with ObfFile(file, mode='r+b', memmap=memmap, squeeze=True) as obf:
        repr(obf)
        str(obf)
        if filetype is str:
            assert obf.filename == str(filename.name)
            assert obf.dirname == str(filename.parent)
        assert not obf.filehandle.closed
        assert obf.name == 'Test.obf'

        header = obf.header
        assert isinstance(header, ObfFileHeader)
        assert header.format_version == 2
        assert header.description == ''
        assert header.first_stack_pos == 34
        assert header.meta_data_position == 189219
        assert list(header.metadata.keys()) == [
            'ome_xml',
            'stedycon_acquisition_begin',
            'stedycon_acquisition_end',
            'stedycon_calibration',
            'stedycon_hardware_configuration',
            'stedycon_sessionstring',
        ]
        assert header.metadata['ome_xml'].startswith(
            '<?xml version="1.0" encoding="UTF-8"?>'
        )

        stacks = obf.stacks
        assert isinstance(stacks, ObfStackSequence)
        assert len(stacks) == 7

        for stack in stacks:
            assert isinstance(stack, ObfStack)
            assert stack is stacks.find(stack.name)
            repr(stack)
            str(stack)

        assert stacks.find('nonexistent') is None
        assert stacks.find('nonexistent', default='x') == 'x'
        assert len(stacks.findall('Confocal')) == 4
        assert len(stacks.findall('nonexistent')) == 0
        with pytest.raises(KeyError):
            stacks['nonexistent']
        with pytest.raises(IndexError):
            stacks[99]

        stack = stacks[0]
        assert stack.name == 'Abberior STAR RED.Confocal'
        assert stack.dtype == numpy.dtype('int16')
        assert stack.shape == (18, 2, 339, 381)
        assert stack.dims == ('T', 'Z', 'Y', 'X')
        assert stack.ndim == 4
        assert stack.sizes == {'T': 18, 'Z': 2, 'Y': 339, 'X': 381}
        assert set(stack.coords.keys()) == {'T', 'Z', 'Y', 'X'}

        shdr = stack.header
        assert isinstance(shdr, ObfStackHeader)
        assert shdr.format_version == 6
        assert shdr.rank == 4
        assert shdr.res == (381, 339, 2, 18)
        assert_allclose(
            shdr.lengths, (7.626657417e-05, 6.78050658e-05, 5e-07, 0.0)
        )
        assert_allclose(shdr.offsets, (0.0, 0.0, 1.25e-07, 0.0))
        assert shdr.data_type == 8
        assert shdr.compression_type == 1
        assert shdr.compression_level == 5
        assert shdr.name == 'Abberior STAR RED.Confocal'
        assert shdr.description == ''
        assert shdr.data_pos == 428
        assert shdr.data_length == 149454
        assert shdr.next_stack_pos == 428

        sftr = stack.footer
        assert isinstance(sftr, ObfStackFooter)
        assert sftr.labels == ['X', 'Y', 'Z', 'T']
        assert sftr.has_col_positions == (0, 0, 0, 0)
        assert sftr.num_flush_points == 1
        assert sftr.flush_block_size == 258318
        assert sftr.stack_end_disk == 152782
        assert sftr.stack_end_used_disk == 152782
        assert sftr.min_format_version == 6
        assert sftr.num_chunk_positions == 3
        assert sftr.samples_written == 203073
        assert isinstance(sftr.si_value, ObfSiUnit)
        assert str(sftr.si_value) == ''
        assert sftr.si_value.scale_factor == 1.0
        assert sftr.si_value.meters[0] == 0
        si_dims = sftr.si_dimensions
        assert all(isinstance(s, ObfSiUnit) for s in si_dims)
        assert [str(s) for s in si_dims[: shdr.rank]] == ['m', 'm', 'm', 's']
        assert si_dims[0].meters == (1, 1)  # X -> m
        assert si_dims[0].scale_factor == 1.0
        assert si_dims[3].seconds == (1, 1)  # T -> s
        assert list(sftr.tag_dictionary.keys()) == ['svi']

        attrs = stack.attrs
        assert attrs['description'] == ''
        assert attrs['si_value'] == ''
        assert attrs['si_dimensions'] == ('s', 'm', 'm', 'm')
        assert list(attrs['tag_dictionary'].keys()) == ['svi']

        data = stack.asxarray()
        assert isinstance(data, DataArray)
        assert data.shape == stack.shape
        assert data.dtype == stack.dtype
        assert data.data.sum(dtype=numpy.uint32) == 89026

        if filetype is str:
            obf.close()
            assert obf.filehandle.closed

    if filetype is not str:
        file.close()
    else:
        with pytest.raises(ValueError):
            obf = ObfFile(file, mode='abc')


@pytest.mark.parametrize('memmap', [False, True])
def test_obf_uncompressed(memmap):
    """Test OBF file v6 uncompressed."""
    filename = DATA / 'openmicroscopy.org/test-v6-uncompressed.obf'

    with ObfFile(filename, squeeze=True, memmap=memmap) as obf:
        repr(obf)
        str(obf)
        assert obf.filename == str(filename.name)
        assert obf.dirname == str(filename.parent)
        assert not obf.filehandle.closed
        assert obf.name == 'test-v6-uncompressed.obf'

        header = obf.header
        assert isinstance(header, ObfFileHeader)
        assert header.format_version == 2
        assert header.description == ''
        assert header.first_stack_pos == 34
        assert header.meta_data_position == 37991829
        assert list(header.metadata.keys()) == [
            'ome_xml',
            'stedycon_acquisition_begin',
            'stedycon_acquisition_end',
            'stedycon_calibration',
            'stedycon_hardware_configuration',
            'stedycon_sessionstring',
        ]
        assert header.metadata['ome_xml'].startswith(
            '<?xml version="1.0" encoding="UTF-8"?>'
        )

        stacks = obf.stacks
        assert isinstance(stacks, ObfStackSequence)
        assert len(stacks) == 4

        for stack in stacks:
            assert isinstance(stack, ObfStack)
            assert stack is stacks.find(stack.name)
            repr(stack)
            str(stack)

        stack = stacks[0]
        assert stack.name == 'Abberior STAR RED.Confocal'
        assert stack.dtype == numpy.dtype('int16')
        assert stack.shape == (18, 2, 339, 381)
        assert stack.dims == ('T', 'Z', 'Y', 'X')
        assert stack.ndim == 4
        assert stack.sizes == {'T': 18, 'Z': 2, 'Y': 339, 'X': 381}
        assert set(stack.coords.keys()) == {'T', 'Z', 'Y', 'X'}

        shdr = stack.header
        assert isinstance(shdr, ObfStackHeader)
        assert shdr.format_version == 6
        assert shdr.rank == 4
        assert shdr.res == (381, 339, 2, 18)
        assert_allclose(
            shdr.lengths, (7.626657417e-05, 6.78050658e-05, 5e-07, 0.0)
        )
        assert_allclose(shdr.offsets, (0.0, 0.0, 1.25e-07, 0.0))
        assert shdr.data_type == 8
        assert shdr.compression_type == 0
        assert shdr.compression_level == 0
        assert shdr.name == 'Abberior STAR RED.Confocal'
        assert shdr.description == ''
        assert shdr.data_pos == 428
        assert shdr.data_length == 36681572
        assert shdr.next_stack_pos == 428

        sftr = stack.footer
        assert isinstance(sftr, ObfStackFooter)
        assert sftr.labels == ['X', 'Y', 'Z', 'T']
        assert sftr.has_col_positions == (0, 0, 0, 0)
        assert sftr.num_flush_points == 0
        assert sftr.flush_block_size == 258318
        assert sftr.stack_end_disk == 36880108
        assert sftr.stack_end_used_disk == 36880108
        assert sftr.min_format_version == 6
        assert sftr.num_chunk_positions == 12204
        assert sftr.samples_written == 4649724
        assert str(sftr.si_value) == ''
        assert [str(s) for s in sftr.si_dimensions[: shdr.rank]] == [
            'm',
            'm',
            'm',
            's',
        ]
        assert list(sftr.tag_dictionary.keys()) == ['svi']

        attrs = stack.attrs
        assert attrs['description'] == ''
        assert attrs['si_value'] == ''
        assert attrs['si_dimensions'] == ('s', 'm', 'm', 'm')
        assert list(attrs['tag_dictionary'].keys()) == ['svi']

        data = stack.asxarray()
        assert isinstance(data, DataArray)
        assert data.shape == stack.shape
        assert data.dtype == stack.dtype
        assert data.data.sum(dtype=numpy.uint32) == 6540899


@pytest.mark.parametrize('memmap', [False, True])
def test_obf_v7(memmap):
    """Test OBF file v7."""
    filename = DATA / 'image.sc-65820/2x Confocal STED .obf'

    with ObfFile(filename, mode='r+b', squeeze=True, memmap=memmap) as obf:
        repr(obf)
        str(obf)
        assert obf.filename == str(filename.name)
        assert obf.dirname == str(filename.parent)
        assert not obf.filehandle.closed
        assert obf.name == '2x Confocal STED .obf'

        header = obf.header
        assert isinstance(header, ObfFileHeader)
        assert header.format_version == 2
        assert header.description == ''
        assert header.first_stack_pos == 34
        assert header.meta_data_position == 3184392
        assert list(header.metadata.keys()) == ['ome_xml']
        assert header.metadata['ome_xml'].startswith(
            '<?xml version="1.0" encoding="UTF-8"?>'
        )

        stacks = obf.stacks
        assert isinstance(stacks, ObfStackSequence)
        assert len(stacks) == 4

        for stack in stacks:
            assert isinstance(stack, ObfStack)
            assert stack is stacks.find(stack.name)
            repr(stack)
            str(stack)

        stack = stacks[0]
        assert stack.name == 'Overview 2-G1-Image 2-STAR RED'
        assert stack.dtype == numpy.dtype('int16')
        assert stack.shape == (18, 282, 201)
        assert stack.dims == ('Z', 'Y', 'X')
        assert stack.ndim == 3
        assert stack.sizes == {'Z': 18, 'Y': 282, 'X': 201}
        assert set(stack.coords.keys()) == {'Z', 'Y', 'X'}

        shdr = stack.header
        assert isinstance(shdr, ObfStackHeader)
        assert shdr.format_version == 7
        assert shdr.rank == 5
        assert shdr.res == (201, 282, 18, 1, 1)
        assert_allclose(
            shdr.lengths,
            (6.02999989548e-06, 8.4599996616e-06, 4.5000001584e-06, 1.0, 1.0),
        )
        assert_allclose(shdr.offsets, (-0.5, -0.5, -0.5, -0.5, -0.5))
        assert shdr.data_type == 8
        assert shdr.compression_type == 1
        assert shdr.compression_level == 1
        assert shdr.name == 'Overview 2-G1-Image 2-STAR RED'
        assert shdr.description == ''
        assert shdr.data_pos == 432
        assert shdr.data_length == 942970
        assert shdr.next_stack_pos == 945098

        sftr = stack.footer
        assert isinstance(sftr, ObfStackFooter)
        assert sftr.labels == ['', '', '', '', '']
        assert sftr.has_col_positions == (0, 0, 0, 0, 0)
        assert sftr.num_flush_points == 18
        assert sftr.flush_block_size == 113364
        assert sftr.stack_end_disk == 945098
        assert sftr.stack_end_used_disk == 945098
        assert sftr.min_format_version == 0
        assert sftr.num_chunk_positions == 0
        assert sftr.samples_written == 1020276
        assert str(sftr.si_value) == ''
        assert [str(s) for s in sftr.si_dimensions[: shdr.rank]] == [
            'm',
            'm',
            'm',
            '',
            '',
        ]
        assert list(sftr.tag_dictionary.keys()) == []

        attrs = stack.attrs
        assert attrs['description'] == ''
        assert attrs['si_value'] == ''
        assert attrs['si_dimensions'] == ('m', 'm', 'm')
        assert 'tag_dictionary' not in attrs

        data = stack.asxarray()
        assert isinstance(data, DataArray)
        assert data.shape == stack.shape
        assert data.dtype == stack.dtype
        assert data.data.sum(dtype=numpy.uint32) == 38935022


@pytest.mark.parametrize('memmap', [False, True])
def test_obf_lifetime(memmap):
    """Test OBF file with lifetime data."""
    filename = DATA / 'zenodo-17039369/IMG0005_Lifetime3.obf'

    with ObfFile(filename, mode='r+b', squeeze=True, memmap=memmap) as obf:
        repr(obf)
        str(obf)
        assert obf.filename == str(filename.name)
        assert obf.dirname == str(filename.parent)
        assert not obf.filehandle.closed
        assert obf.name == 'IMG0005_Lifetime3.obf'

        header = obf.header
        assert isinstance(header, ObfFileHeader)
        assert header.format_version == 2
        assert header.description == ''
        assert header.first_stack_pos == 34
        assert header.meta_data_position == 571045
        assert list(header.metadata.keys()) == [
            'ome_xml',
            'stedycon_acquisition_end',
            'stedycon_hardware_configuration',
            'stedycon_sessionstring',
            'stedycon_calibration',
            'stedycon_acquisition_begin',
        ]
        assert header.metadata['ome_xml'].startswith('<OME ')

        stacks = obf.stacks
        assert isinstance(stacks, ObfStackSequence)
        assert len(stacks) == 6

        for stack in stacks:
            assert isinstance(stack, ObfStack)
            assert stack is stacks.find(stack.name)
            repr(stack)
            str(stack)

        stack = stacks[-1]
        assert stack.name == 'STAR GREEN.Confocal.phasor'
        assert stack.dtype == numpy.complex64
        assert stack.shape == (181, 200)
        assert stack.dims == ('Y', 'X')
        assert stack.ndim == 2
        assert stack.sizes == {'Y': 181, 'X': 200}
        assert set(stack.coords.keys()) == {'Y', 'X'}

        shdr = stack.header
        assert isinstance(shdr, ObfStackHeader)
        assert shdr.format_version == 7
        assert shdr.rank == 2
        assert shdr.res == (200, 181)
        assert_allclose(shdr.lengths, (2e-05, 1.81e-05))
        assert_allclose(shdr.offsets, (0.0, 0.0))
        assert shdr.data_type == 1073741888
        assert shdr.compression_type == 1
        assert shdr.compression_level == 5
        assert shdr.name == 'STAR GREEN.Confocal.phasor'
        assert shdr.description == ''
        assert shdr.data_pos == 2372
        assert shdr.data_length == 566372
        assert shdr.next_stack_pos == 0

        sftr = stack.footer
        assert isinstance(sftr, ObfStackFooter)
        assert sftr.labels == ['X', 'Y']
        assert sftr.has_col_positions == (0, 0)
        assert sftr.num_flush_points == 1
        assert sftr.flush_block_size == 289600
        assert sftr.stack_end_disk == 571045
        assert sftr.stack_end_used_disk == 571045
        assert sftr.min_format_version == 0
        assert sftr.num_chunk_positions == 1
        assert sftr.samples_written == 36200
        assert str(sftr.si_value) == ''
        assert [str(s) for s in sftr.si_dimensions[: shdr.rank]] == ['m', 'm']
        assert list(sftr.tag_dictionary.keys()) == ['svi']

        attrs = stack.attrs
        assert attrs['description'] == ''
        assert attrs['si_value'] == ''
        assert attrs['si_dimensions'] == ('m', 'm')
        assert list(attrs['tag_dictionary'].keys()) == ['svi']

        data = stack.asxarray()
        assert isinstance(data, DataArray)
        assert data.shape == stack.shape
        assert data.dtype == stack.dtype
        assert_allclose(
            data.data.sum(dtype=numpy.complex128),
            13602.964329123497 + 15436.518050804734j,
            rtol=1e-5,
        )


@pytest.mark.skipif(
    not hasattr(sys, '_is_gil_enabled'), reason='Python < 3.13'
)
def test_gil_enabled():
    """Test that GIL state is consistent with build configuration."""
    assert sys._is_gil_enabled() != sysconfig.get_config_var('Py_GIL_DISABLED')


@pytest.mark.parametrize(
    'fname',
    tuple(
        itertools.chain.from_iterable(
            glob.glob(f'**/*{ext}', root_dir=DATA, recursive=True)
            for ext in FILE_EXTENSIONS
        )
    ),
)
def test_glob(fname):
    """Test read all OBF files."""
    if 'defective' in fname:
        pytest.xfail(reason='file is marked defective')
    fname = DATA / fname
    with ObfFile(fname) as obf:
        str(obf)
        for stack in obf.stacks:
            str(stack)
            data = stack.asxarray()
            assert data.shape == stack.shape
            assert data.dtype == stack.dtype


if __name__ == '__main__':
    import warnings

    # warnings.simplefilter('always')
    warnings.filterwarnings('ignore', category=ImportWarning)
    argv = sys.argv
    argv.append('--cov-report=html')
    argv.append('--cov=obffile')
    argv.append('--verbose')
    sys.exit(pytest.main(argv))

# mypy: allow-untyped-defs
# mypy: check-untyped-defs=False
