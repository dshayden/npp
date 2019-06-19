from setuptools import setup


def readme():
  with open('README.rst') as f:
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
  install_requires=[
    'numpy', 'scipy', 'lie', 'matplotlib', 'scikit-learn', 'trimesh'
  ], # scipy-stats?
  test_suite='nose.collector',
  tests_require=['nose', 'nose-cover3', 'numdifftools'],
  # extras = {
  #   'test': tests_require,
  # },
  include_package_data=True,
  zip_safe=False)
