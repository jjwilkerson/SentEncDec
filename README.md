# SentEncDec

## Overview
SentEncDec is a specialized module focused on encoding and decoding sentences into vector representations. This module stands out for its two main innovations: a unique residual-recurrent architecture and the match drop technique.

The residual-recurrent architecture allows the network to be efficiently trained using first-order optimization algorithms, such as ADAM. This approach enhances the learning process, making it more effective and streamlined, especially in handling the complexities of natural language. This architecture can also be used in other tasks that use neural networks to process sequential data.

In addition, the match drop technique is a key feature of SentEncDec. It smartly allocates computational resources to the most crucial parts of the output. This targeted approach ensures that the network's capacity is utilized where it's needed the most, leading to more efficient processing and better performance.

These advancements make SentEncDec a valuable tool for various applications in natural language processing, from semantic analysis to more complex tasks like language translation. Whether for academic research or practical industry applications, SentEncDec offers a robust, innovative solution for sentence encoding and decoding.

This project is associated with the paper titled "Return of the RNN: Residual Recurrent Networks for Invertible Sentence Embeddings" which provides in-depth explanations of the concepts, methodologies, and findings that underpin this software. The paper can be accessed [here](https://arxiv.org/abs/2303.13570v2).

For detailed documentation, see [SentEncDec Documentation](https://jjwilkerson.github.io/SentEncDec/).

## Compatibility

This software has been developed and tested on Linux. While it may work on other UNIX-like systems, its compatibility with non-Linux operating systems (like Windows or macOS) has not been verified. Users are welcome to try running it on other systems, but should be aware that they may encounter issues or unexpected behavior.

## Dependencies
To build and use SentEncDec, you'll need the following dependencies:

- **C++ Compiler:** GCC (versions 6.x - 12.2) or an equivalent compiler.
- **CUDA Toolkit:** Version 12.1 - [Download CUDA Toolkit](https://developer.nvidia.com/cuda-downloads).
- **Additional Libraries:**
    - NetLib (another module next to this one).
    - UTF8-CPP (version 2.3.4) [Download UTF8-CPP](https://github.com/nemtrif/utfcpp/tree/v2.3.4).
    - JsonCpp (version 1.8.4) [Download JsonCpp](https://github.com/open-source-parsers/jsoncpp/tree/1.8.4).
    - Boost (version 1.72.0) [Download Boost](https://www.boost.org/users/history/).
        - Build the regex, system, and filesystem modules.
    - CUDA Samples Common [Download CUDA Samples](https://github.com/NVIDIA/cuda-samples).

Ensure you have these libraries installed and accessible in your development environment.

## Configuring Dependencies in Eclipse
If building with Eclipse, it is necessary to configure the dependencies. This involves setting up include paths, library paths, and linking the libraries. Here's how to do it:

### Include Paths
- Add the include paths for NetLib, UTF8-CPP, and CUDA Samples Common
- Example paths:
    - ${workspace_loc:/NetLib}
    - /usr/local/include/utf8
    - /usr/local/cuda-samples/Common

### Library Paths
- Specify the library paths where Eclipse can find the compiled libraries.
- Example paths:
    - ${workspace_loc:/NetLib/Release}
    - /usr/local/lib (location of JsonCpp library)
    - /usr/lib/x86_64-linux-gnu (location of Boost libraries
    - /usr/local/cuda-12.1/lib64 (location of CUDA libraries

### Libraries
- Link the libraries in your Eclipse project settings.
- Example libraries to link:
    - NetLib
    - boost_regex
    - boost_system
    - boost_filesystem
    - jsoncpp
    - cublas

## Building SentEncDec
While SentEncDec was developed using Eclipse with the Nsight plugin, it's not a strict requirement. You can build it as long as you have the CUDA toolkit installed.

Here are the general steps to build SentEncDec:

- **Clone the Repository:**
 
```bash
git clone https://github.com/jjwilkerson/SentEncDec.git
cd SentEncDec
```

- **Building:**
If the above include paths, library paths, and libraries are configured then SentEncDec can be easily built (as an executable) in Eclipse with the Nsight plugin. It is necessary to set the PROGRAM_VERSION macro to a value from 1 to 5 to determine which executable program is built.

| PROGRAM_VERSION | Program                                             |
|:---------------:|:--------------------------------------------------- |
|        1        | Single training run                                 |
|        2        | Random search for hyperparameters                   |
|        3        | Extend a training run (such as after random search) |
|        4        | Encoder (inference)                                 |
|        5        | Decoder (inference)                                 |

It is also necessary to set the NETWORK_VERSION macro to a value of 1 or 2 to build a residual-recurrent or GRU network, respectively.

Alternatively, you can build it from the command line using Make by executing one of the following commands. It may be necessary to update paths to nvcc and dependencies in Makefile first. You may also need to change the CUDA hardware versions after "arch=" to match your specific GPU.

Single training run, residual-recurrent:

```bash
make
```

or 

```bash
make rr
```

Single training run, GRU:

```bash
make gru
```

Random search for hyperparameters, residual-recurrent:

```bash
make search-rr
```

Random search for hyperparameters, GRU:

```bash
make search-gru
```

Extend a training run (such as after random search), residual-recurrent:

```bash
make extend-rr
```

Extend a training run, GRU:

```bash
make extend-gru
```

Encoder (inference): 

```bash
make encoder
```

Decoder (inference): 

```bash
make decoder
```

Before building a different target, run the following:

```bash
make clean
```

## Usage
Before running the program, SED_DATASETS_DIR must be defined as an environment variable, with its value being the location of the SentEncDec datasets in your file system. The datasets can be downloaded from the following link: [Download datasets](https://drive.google.com/drive/folders/1A-t6bfeG3_HzQt8Cvsbijk-RPqPccx7g?usp=sharing). Decompress the tar file before using.

To run as a single training run, place the executable in a new folder along with a config file named config.json. You can use the config.json file in the SentEncDec source folder as a starting point. The program state and parameters will be saved periodically, so that training can be restarted if interrupted.

To run as a random search, a config file will not be necessary. To extend a training run, place the executable in the folder corresponding to the training run and run it from there.

To run inference (encoder or decoder), place the executable in a folder with a config file and saved weights (parameters) and run it from there.

## Contributing
Contributions to SentEncDec are welcome. Please ensure to follow the coding standards and submit a pull request for review.

## License
SentEncDec is licensed under the MIT License. See the LICENSE file for more details.
