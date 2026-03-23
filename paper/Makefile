MAIN = main

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex references.bib neurips_2026.sty
	pdflatex $(MAIN)
	bibtex $(MAIN)
	pdflatex $(MAIN)
	pdflatex $(MAIN)

clean:
	rm -f $(MAIN).{aux,bbl,blg,log,out,pdf,synctex.gz,fls,fdb_latexmk}

.PHONY: all clean
