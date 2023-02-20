"""
If you run this to build the hull extensions you need to have Cython installed and a C/C++
compiler. If you dont know how to recompile it all then take at look at the Cython website.

I personally find it all a little confusing :)

However, when and if you do run this run it as
> python setup.py bdist to get the .c and .pyd files. These are what python will use when you install the main package
"""
from setuptools import setup, Extension
from Cython.Build import cythonize

compiler_directives = {"language_level": 3, "embedsignature": True}
extensions = [Extension("chulls", ["chulls.pyx"])]
extensions = cythonize(extensions, compiler_directives=compiler_directives)
setup(
    name="chulls Test",
    # package_dir={r"C:\2022\SpectralTools\spectraltools\ext\\": ''},
    zip_safe=False,
    ext_modules=cythonize(extensions),  # , compiler_directives)
)
