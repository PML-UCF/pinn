import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='pml-pinn',
    version="0.0.2",
    author='Felipe Viana, Renato G. Nascimento, Yigit Yucesan, Arinan Dourado',
    author_email='viana@ucf.edu, renato.gn@knights.ucf.edu, yucesan@knights.ucf.edu, arinandourado@knights.ucf.edu',
    description='Physics-informed neural networks',
    url='https://github.com/PML-UCF/pinn',
    license='MIT',
    long_description=read('README.md'),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=['numpy', 'tensorflow'],
    keywords=[
        'physics informed',
        'neural networks',
        'machine learning',
        'deep learning',
        'tensorflow',
        'keras',
        'python'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6',
        'Topic :: Education',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ]
)
