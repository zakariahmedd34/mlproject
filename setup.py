from setuptools import find_packages, setup
from typing import List
HYPHEN_E_DOT = '-e .' 
def get_requirements(file_path: str) ->List[str]:
    '''
    Args:
        file_path (str): path to requirements.txt file
    Returns:
        List[str]: list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","")for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements

setup(
    name = "mlproject",
    version = "0.0.1",
    author = "Zakaria",
    author_email = "zakariahmed870@gmail.com",
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
)