# Makefile for LaTeX paper compilation

# Main document
MAIN = main
SECTIONS = sections/abstract.tex sections/introduction.tex sections/methods.tex \
           sections/experiments.tex sections/results.tex sections/discussion.tex \
           sections/conclusion.tex
REFS = references.bib

# LaTeX compiler
LATEX = pdflatex
BIBTEX = bibtex

# Targets
.PHONY: all clean view

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex $(SECTIONS) $(REFS)
	$(LATEX) $(MAIN)
	$(BIBTEX) $(MAIN)
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)

clean:
	rm -f *.aux *.log *.bbl *.blg *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz
	rm -f $(MAIN).pdf

view: $(MAIN).pdf
	xdg-open $(MAIN).pdf 2>/dev/null || open $(MAIN).pdf 2>/dev/null || echo "Please open $(MAIN).pdf manually"

# Watch mode (requires latexmk)
watch:
	latexmk -pvc -pdf $(MAIN)

# Quick compile (no bibtex)
quick:
	$(LATEX) $(MAIN)

help:
	@echo "Available targets:"
	@echo "  all     - Build the full paper (default)"
	@echo "  clean   - Remove generated files"
	@echo "  view    - Build and open the PDF"
	@echo "  quick   - Quick compile without bibtex"
	@echo "  watch   - Continuous compilation (requires latexmk)"
	@echo "  help    - Show this message"
