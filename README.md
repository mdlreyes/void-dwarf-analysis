# void-dwarf-analysis

<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About](#about)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Contact](#contact)
* [Acknowledgements](#acknowledgements)



<!-- ABOUT THE PROJECT -->
## About

This is a pipeline for producing maps of kinematic properties (velocity and velocity dispersion), emission line fluxes, and gas-phase metallicities from KCWI datacubes.

Before running this analysis pipeline, these datacubes must be first be reduced by the [KCWI Data Reduction Pipeline](https://github.com/Keck-DataReductionPipelines/KcwiDRP). You may also want to stack multiple exposures; I used [this code](https://github.com/yuguangchen1/kcwi) by Yuguang Chen to do this. 


<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

Here are the main Python packages required and the versions used:
* astropy (4.0)
* cwitools (0.7)
* matplotlib (3.2.1)
* numpy (1.18.2)
* pandas (1.0.3)
* ppxf (7.0.1)
* vorbin (3.1.4)

Note that this is **not** an exhaustive list! The easiest way to install the full list of required packages is to create a conda environment using the enclosed `kcwiredux_env.yml` file:
```sh
conda env create -f kcwiredux_env.yml
```
Note that this file lists v0.6 of cwitools, but functions from v0.7 are needed. You may want to clone directly from Donal O'Sullivan's [github repository](https://github.com/dbosul/cwitools).

### Installation

Follow the usual steps to clone the repo:
```sh
git clone https://github.com/mdlreyes/void-dwarf-analysis.git
```


<!-- USAGE EXAMPLES 
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_
-->

<!-- CONTRIBUTING 
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
-->


<!-- CONTACT -->
## Contact

Mia de los Reyes - [@MiaDoesAstro](https://twitter.com/MiaDoesAstro) - mdelosre@caltech.edu

Project Link: [https://github.com/mdlreyes/void-dwarf-analysis](https://github.com/mdlreyes/void-dwarf-analysis)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

[README.md template](https://github.com/othneildrew/Best-README-Template)
