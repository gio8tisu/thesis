# Index

1. Introduction [here](1-introduction/text.md)

	* Project Background
	* Need
	* Problem specification
	* Techniques
	* Work Structure

2. Atificial Neural Networks [here](2-theoric_background/text.md)

	* Perceptron
	* Multi-layer perceptron (MLP)
	* Convolutional Neural Networks (CNN)
		- Fully-convolutional Networks
	* Generative Adversarial Networks (GANs)
	* GANs Proof of Concept
	* GANs for image-to-image translation
		- Pix2Pix
		- CycleGANs

3. Despeckling network [here](3-methodology/text.md)

	* Speckle noise
	* Proposed network architectures

4. Stain Network [here](4-experiments_and_results/text.md)

	* Proposed architecture

5. Result validation [here](5-conclusions_and_future_development/text.md)

	* Experiments
	* Histology Professionals Validation

-----------------------------------------------

# README

`make` to build `main.pdf` (thesis document).  
`make clean` to remove "garbage" files.

## LaTeX project divided into multiple files
Thesis main sections/chapters are divided into sub-directories and `main.tex` "grabs" them using subfiles package.  
The document preamble and glossary are also in separate files (`preamble.sty` and `acronyms.tex`).  
Bibliography is in BibTeX file (`refs.bib`).
