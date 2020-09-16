from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name="av-goal-recogntion",
      author="Cillian Brewitt",
      version='1.0.0',
      packages=['av_goal_recognition'],
      install_requires=requirements)
