from setuptools import find_packages ,setup
from typing import List


Hyphen_e = '-e .'
def get_requirements(file_path:str)->List[str]:
    ''''
    This function returnt the list of requirements

    '''
    requirements = []
    with open (file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("/n","") for req in requirements]
        
        if Hyphen_e in requirements:
            requirements.remove(Hyphen_e)

    return requirements




setup (
    name = 'ml_project',
    version = '0.0.1',
    author = 'Prajwal',
    author_email = 'prajwalkuhite50@gmail.com',
    packages = find_packages(),
    install_requires =get_requirements('requirements.txt')

    
)