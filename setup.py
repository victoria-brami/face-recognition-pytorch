from setuptools import setup, find_packages

setup(
    name='facenet',
    version='1.0',
    author='Victoria Brami',
    url='https://github.com/victoria-brami/face-recognition-pytorch.git',
    install_requires=["pytorch-lightning", "hydra-core"],
    packages=find_packages(),
)