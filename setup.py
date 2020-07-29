import pip
import logging
import pkg_resources
try:
  from setuptools import setup
except ImportError:
  from distutils.core import setup

try:
  install_reqs = _parse_requirements("requirements.txt")
except Exception:
  logging.warning('Fail load requirements file, so using default ones.')
  install_reqs = []

def readme():
  with open('README.md') as f:
    return f.read()

setup(name='npp',
  version='0.1',
  description='Nonparametric Parts',
  long_description=readme(),
  classifiers=[
    'Development Status :: 3 - Alpha',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6.7',
  ],
  keywords='bayesian nonparametric time series',
  url='',
  author='David S. Hayden',
  author_email='dshayden@mit.edu',
  license='MIT',
  packages=['npp'],
  install_requires=install_reqs,
  # install_requires=[
  #   'numpy', 'scipy', 'lie', 'matplotlib', 'scikit-learn', 'trimesh', 'Shapely'
  # ],
  test_suite='nose.collector',
  tests_require=['nose', 'nose-cover3', 'numdifftools'],
  include_package_data=True,
  zip_safe=False)
