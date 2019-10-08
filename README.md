# Index

1. Introduction [here](1-introduction/text.md)

* Project Background
* Need
* Problem specification
* Techniques
* Work Structure

2. Atificial Neural Networks [here](2-theoric_background/text.md)

* Convolutional Neural Networks (CNN)
* Generative Adversarial Networks (GANs)
* GANs for image-to-image translation
	- Pix2Pix
	- CycleGANs

3. Methodology [here](3-methodology/text.md)

3.1 Despeckling network

* Speckle noise
* Proposed network architectures

3.2 Stain Network

* Proposed architecture

4. Experiments and results [here](4-experiments_and_results/text.md)
	
4.1 GAN PoC

4.2 Despeckling network

* Experiments
* Results

4.3 Stain network

* Experiments
* Results

5. Result validation [here](5-conclusions_and_future_development/text.md)

* Experiments
* Histology Professionals Validation

-----------------------------------------------

# README

## Makefile
`make all` to build `main.pdf` file (thesis document).  
`make clean` to remove "garbage" files.

## LaTeX project divided into multiple files
Thesis main sections/chapters are divided into files inside sub-directories and `main.tex` "grabs" them using subfiles package.  
The document preamble and glossary are also in separate files (`preamble.sty` and `acronyms.tex`).  
Bibliography is in BibTeX file (`refs.bib`).
