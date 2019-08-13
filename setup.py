from setuptools import setup

setup(name='rl',
      author='Tim Vieira',
      description='Reference implementation of algorithms for reinforcement learning and Markov decision processes.',
      version='1.0',
      install_requires=[
          'arsenal',
          'numpy',
          'scipy',
          'pandas',
      ],
      dependency_links=[
          'https://github.com/timvieira/arsenal.git',
      ],
      packages=['rl'],
)
