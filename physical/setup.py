
from distutils.core import setup
import warnings

setup(name='hytoperm',
      version='0.1',
      python_requires='>=3.8',
      description='Hybrid Trajectory Optimizer for Persistent Monitoring',
      author='Jonas Hall',
      license='MIT',
      include_package_data=True,
      py_modules=[],
      setup_requires=['setuptools_scm'],
      install_requires=[
          'matplotlib',
          'numpy>=1.24.4,<2.0.0',
          'casadi>=3.6.3',
          'pdfCropMargins>=2.1.1',
          'scipy',
          'python-tsp>=0.4.0',
      ]
      )

warnings.filterwarnings("always")
