SHELL=/bin/bash
PREFIX=presentation


.PHONY: all
all : $(PREFIX).aux $(PREFIX).log $(PREFIX).out $(PREFIX).toc $(PREFIX).tex
	pdflatex $(PREFIX)

$(PREFIX).aux $(PREFIX).log $(PREFIX).out $(PREFIX).toc : $(PREFIX).tex
	pdflatex $(PREFIX)

.PHONY: clean
clean :  ## Clean output files
	@rm $(PREFIX).{aux,log,out,toc}
