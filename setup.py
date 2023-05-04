from setuptools import find_packages, setup
from typing import List

hyphen_e_dot = "-e ."

def get_requirements(file_path:str)->List[str]:
    requirements = []
    with open(file_path) as file_object:
        requirement = file_object.readlines()
        requirements = [i.replace("\n","") for i in requirement]

        if hyphen_e_dot in requirements:
            requirements.remove(hyphen_e_dot)

    return requirements


setup(
    name='RegressorProject',
    version='0.0.1',
    author='Ajit',
    author_email='ajitk2805@gmail.com',
    install_requires=get_requirements("requirements.txt"),
    packages=find_packages()

)