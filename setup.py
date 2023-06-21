from setuptools import setup

setup(
    name='advdiff_mcmc',
    version='0.1.2',    
    description='A Python package for Bayesian estimation of parameters in an advection-diffusion equation.',
    url='https://github.com/williamedge/advdiff_mcmc',
    author='William Edge',
    author_email='william.edge@uwa.edu.au',
    license='BSD 3-clause',
    packages=['advdiff_mcmc'],
    install_requires=['numpy',
                      'scipy',
                      'pymc'],

    classifiers=[
        'Development Status :: 2 - Improvement',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD 3-Clause License',  
        'Operating System :: POSIX :: All?',        
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
    ],
)