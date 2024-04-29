# Test linting of project

import sys
import os
from io import StringIO
import unittest
from glob import glob
import pycodestyle


class TestCodeFormat(unittest.TestCase):

    def test_conformance(self):
        """Test that we conform to PEP-8."""
        project_dir = os.path.dirname(os.path.realpath('setup.py'))
        file_paths = glob(
            os.path.join(project_dir, 'resurfemg', '**/*.py'),
            recursive=True,
            ) + [os.path.join(project_dir, 'setup.py')]
        
        buffer = StringIO()
        sys.stdout = buffer
        
        style = pycodestyle.StyleGuide()
        result = style.check_files(file_paths)

        print_output = buffer.getvalue()
        sys.stdout = sys.__stdout__

        self.assertEqual(result.total_errors, 0,
                         "Found code style errors (and warnings):\n"
                         + print_output)