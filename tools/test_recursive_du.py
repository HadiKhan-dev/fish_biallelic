import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location('recursive_du', str(Path(__file__).with_name('recursive_du.py')))
du = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(du)


BACKENDS = ['python']
try:
    du._NativeBatch()
except OSError:
    pass
else:
    BACKENDS.append('native')


def reference(path):
    return int(subprocess.check_output(['du', '-s', '-B1', str(path)]).split()[0])


class RecursiveDuTests(unittest.TestCase):
    def test_matches_du_with_links_sparse_hidden_and_nested_files(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            base = Path(tmp)
            root = base / 'tree'
            root.mkdir()
            (root / 'nested').mkdir()
            (base / 'outside').write_bytes(b'z' * 8192)
            for i in range(37):
                (root / ('file_%02d' % i)).write_bytes(b'x' * (i * 1024))
            (root / '.hidden').write_bytes(b'hidden')
            os.mkfifo(str(root / 'fifo'))
            with open(os.fsencode(str(root)) + b'/nonutf8_\xff', 'wb') as out:
                out.write(b'non-UTF8 filename')
            with (root / 'sparse').open('wb') as out:
                out.seek(32 * 1024**2)
                out.write(b'x')
            os.link(str(root / 'file_01'), str(root / 'nested' / 'hardlink'))
            (root / 'external').symlink_to(base / 'outside')
            (root / 'loop').symlink_to(root, target_is_directory=True)
            # A hardlinked symlink is also one inode, not two allocations.
            os.link(str(root / 'external'), str(root / 'nested' / 'linked_symlink'), follow_symlinks=False)
            expected = reference(root)
            for backend in BACKENDS:
                for workers, batch_size in [(1, 1), (4, 3), (16, 256)]:
                    result = du.measure(root, workers, batch_size, progress_seconds=0, backend=backend)
                    self.assertTrue(result['complete'], result['errors'])
                    self.assertEqual(expected, result['allocated_bytes'])
                    self.assertEqual(backend, result['backend'])
                for path in (root / 'file_01', root / 'sparse', root / 'external'):
                    self.assertEqual(reference(path), du.measure(path, backend=backend)['allocated_bytes'])

    def test_one_wide_directory_uses_multiple_workers(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            for i in range(32):
                Path(tmp, str(i)).write_bytes(b'x')
            original_stat = os.stat
            barrier = threading.Barrier(4, timeout=10)
            lock = threading.Lock()
            started = [0]
            thread_ids = set()

            def rendezvous(path, **kwargs):
                with lock:
                    started[0] += 1
                    first = started[0] <= 4
                    thread_ids.add(threading.get_ident())
                if first:
                    barrier.wait()
                return original_stat(path, **kwargs)

            with mock.patch.object(du.os, 'stat', side_effect=rendezvous):
                result = du.measure(tmp, workers=4, batch_size=2, progress_seconds=0, backend="python")
            self.assertTrue(result['complete'], result['errors'])
            self.assertEqual(reference(tmp), result['allocated_bytes'])
            self.assertEqual(4, len(thread_ids))

    def test_vanished_file_is_reported_as_partial(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            disappearing = Path(tmp, 'vanishing.tmp')
            disappearing.write_bytes(b'x')
            Path(tmp, 'retained').write_bytes(b'y')
            original_stat = os.stat

            def remove_then_stat(path, **kwargs):
                if path == 'vanishing.tmp':
                    disappearing.unlink()
                return original_stat(path, **kwargs)

            with mock.patch.object(du.os, 'stat', side_effect=remove_then_stat):
                result = du.measure(tmp, workers=2, batch_size=1, progress_seconds=0, backend="python")
            self.assertFalse(result['complete'])
            self.assertEqual(1, result['error_count'])
            self.assertIn('vanishing.tmp', result['errors'][0])
            self.assertEqual(reference(tmp), result['allocated_bytes'])

    @unittest.skipUnless(os.path.isdir('/proc/self/fd'), 'Linux FD accounting')
    def test_directory_descriptors_are_released(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            for i in range(24):
                directory = Path(tmp, str(i))
                directory.mkdir()
                for j in range(5):
                    (directory / str(j)).write_bytes(b'x')
            for backend in BACKENDS:
                before = len(os.listdir('/proc/self/fd'))
                result = du.measure(tmp, workers=8, batch_size=1, progress_seconds=0, backend=backend)
                self.assertTrue(result['complete'], result['errors'])
                self.assertEqual(reference(tmp), result['allocated_bytes'])
                self.assertEqual(before, len(os.listdir('/proc/self/fd')))

    @unittest.skipUnless('native' in BACKENDS, 'optional native helper not built')
    def test_native_vanished_file_is_reported_as_partial(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            disappearing = Path(tmp, 'vanishing.tmp')
            disappearing.write_bytes(b'x')
            Path(tmp, 'retained').write_bytes(b'y')
            original_call = du._NativeBatch.__call__

            def remove_then_scan(helper, fd, names):
                if 'vanishing.tmp' in names:
                    disappearing.unlink()
                return original_call(helper, fd, names)

            with mock.patch.object(du._NativeBatch, '__call__', remove_then_scan):
                result = du.measure(tmp, workers=2, batch_size=1, progress_seconds=0, backend='native')
            self.assertFalse(result['complete'])
            self.assertEqual(1, result['error_count'])
            self.assertIn('vanishing.tmp', result['errors'][0])
            self.assertEqual(reference(tmp), result['allocated_bytes'])

    def test_auto_falls_back_when_native_helper_is_unavailable(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            Path(tmp, 'file').write_bytes(b'x')
            with mock.patch.object(du, '_NativeBatch', side_effect=OSError('helper unavailable')):
                result = du.measure(tmp, progress_seconds=0)
                self.assertEqual('python', result['backend'])
                self.assertTrue(result['complete'])
                self.assertEqual(reference(tmp), result['allocated_bytes'])
                with self.assertRaises(OSError):
                    du.measure(tmp, backend='native')

    def test_cli_json_and_missing_root_exit_status(self):
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
            script = str(Path(__file__).with_name('recursive_du.py'))
            for path, expected_code in [(tmp, 0), (os.path.join(tmp, 'missing'), 1)]:
                proc = subprocess.run([sys.executable, script, path, '--json', '--progress-seconds', '0'],
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
                self.assertEqual(expected_code, proc.returncode, proc.stderr)
                result = json.loads(proc.stdout)
                self.assertEqual(expected_code == 0, result['complete'])
                self.assertIn('finished_at_utc', result)
                self.assertEqual(result['allocated_bytes'] / 1e12, result['TB'])


if __name__ == '__main__':
    unittest.main()
