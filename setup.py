import sys
import textwrap

import pkg_resources
from Cython.Build import cythonize
from setuptools import setup, find_packages, Extension
from setuptools.command.build_py import build_py as _build_py


class build_py(_build_py):
    def run(self):
        self.run_command("build_ext")
        return super().run()

try:
    pkg_resources.require('setuptools >= 18.0')
except pkg_resources.ResolutionError:
    print(textwrap.dedent("""
        setuptools >= 18.0 is required, and the dependency cannot be
        automatically resolved with the version of setuptools that is
        currently installed (%s).

        You can upgrade setuptools:
        $ pip install -U setuptools
        """ % pkg_resources.get_distribution("setuptools").version),
          file=sys.stderr)
    sys.exit(1)

# Here we try to import Cython - if it's here then we can generate new c sources
# directly from the pyx files using their build_ext class.
# If not then we just use the default setuptools version
try:
    from Cython.Distutils import build_ext
    HAVE_CYTHON = True
except ImportError:
    from setuptools.command.build_ext import build_ext
    # from distutils.command.build_ext import build_ext
    HAVE_CYTHON = False

# setting this flag to false for distribution
HAVE_CYTHON = False

# Get extensions and update command class
# CMDCLASS = {
#     'build_ext': build_ext
# }
extensions = {}
ext = '.pyx' if HAVE_CYTHON else '.c'

extensions = [Extension("spex.ext.chulls", ["spex/ext/chulls"+ext])]

if HAVE_CYTHON:
    #extensions = [Extension("spex.ext.chulls", ["spex/ext/chulls.pyx"])]
    compiler_directives = {"language_level": 3, "embedsignature": True}
    extensions = cythonize(extensions, compiler_directives=compiler_directives)

setup(
    name='spex',
    version='0.1.67',
    packages=find_packages(),
    package_data={'spex': ['ext/*.pyd', 'ext/*.so', 'speclib/spectral_libraries/*.csv']},
    include_package_data=True,
    url='',
    author='Andrew Rodger',
    author_email='andrew.rodger@csiro.au',
    license='',
    description='A collection of tools that I have used for working with hyperspectral data',
    install_requires=['numpy', 'cython', 'spectral', 'pandas', 'matplotlib', 'scipy', 'scikit-learn', 'plotly'],
    python_requires='>=3.7',
    setup_requires=['cython', 'numpy', 'wheel'],
    zip_safe=False,
    # Cython extensions & other stuff
    cmdclass={'build_py': build_py, 'build_ext': build_ext},
    ext_modules=extensions
)
