from setuptools import setup, Extension
from Cython.Build import cythonize

compiler_directives = {"language_level": 3, "embedsignature": True}
extensions = [Extension("chulls", ["chulls.pyx"])]
extensions = cythonize(extensions, compiler_directives=compiler_directives)
setup(
    name='chulls Test',
    package_dir={r"C:\2022\SpectralTools\spectraltools\ext\\": ''},
    zip_safe=False,
    ext_modules=cythonize(extensions)#, compiler_directives)
)