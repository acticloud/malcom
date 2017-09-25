figs=../figs/*.eps


all: presentation.pdf

presentation.pdf: presentation.tex introduction.tex experiments.tex
	@pdflatex $<
	#bibtex    presentation
	@pdflatex $<
	@pdflatex $<

clean:
	@rm -f *~ *.nav *.out *.aux *.log *.snm *.toc *.backup *.vrb *.bbl *.blg

distclean: clean
	@rm -f presentation.pdf
