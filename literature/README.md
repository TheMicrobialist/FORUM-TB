# Literature

Papers, preprints and reports read while building FORUM-TB.

**PDFs are not committed.**

| File | Tracked | What |
|---|---|---|
| `background.bib` | yes | everything read, cited or not |
| `*.pdf` | no | local copies, gitignored |
| `../manuscript/references.bib` | yes | only what the manuscript cites |

## Naming

Name the PDFs however you like — they are local, nobody else sees them.

To keep a file findable from its entry, either match the BibTeX key:

    literature/cole1998deciphering.pdf   ->   @article{cole1998deciphering, ...}

or use your own scheme and record it in the entry:

    @article{cole1998deciphering,
      ...
      file = {1998_tb_genome_sequence.pdf}
    }

## Adding a paper

1. Drop the PDF in `literature/`
2. Add the entry to `background.bib` with a DOI or PMID
3. If it later gets cited, move the entry to `manuscript/references.bib`
