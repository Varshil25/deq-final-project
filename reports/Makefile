# Build any report with `make <name>`; `make all` builds everything.
# latexmk runs pdflatex/bibtex as many times as the citations need.

TARGETS = proposal report1 report2 final
LATEXMK = latexmk -pdf -interaction=nonstopmode -halt-on-error

.PHONY: all clean $(TARGETS)

all: $(TARGETS)

$(TARGETS): %:
	$(LATEXMK) $*.tex
	@echo "--- $*.pdf: $$(pdfinfo $*.pdf | awk '/^Pages/{print $$2}') page(s) ---"

# Page limits: proposal 2, report1 2, report2 2, final 5.
check: all
	@for t in $(TARGETS); do \
	  p=$$(pdfinfo $$t.pdf | awk '/^Pages/{print $$2}'); \
	  echo "$$t.pdf: $$p page(s)"; \
	done

clean:
	latexmk -C
	rm -f *.bbl *.blg *.aux *.log *.out *.fls *.fdb_latexmk
