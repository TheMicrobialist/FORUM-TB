# Literature

Papers, preprints and reports read while building FORUM-TB.

**PDFs are not committed.** Most are copyrighted and redistributing them in a
public repo is not ours to do, and git would keep every version forever in
every clone. They stay local; the metadata is tracked so anyone can fetch
their own copy from the DOI.

| File | Tracked | What |
|---|---|---|
| `background.bib` | yes | everything read, cited or not |
| `*.pdf` | no | local copies, gitignored |
| `../manuscript/references.bib` | yes | only what the manuscript cites |

## Convention

**Name each PDF after its BibTeX key**, so the file and the entry are one
lookup apart:

    literature/cole1998deciphering.pdf   ->   @article{cole1998deciphering, ...}

Keys follow the style already in `references.bib`: first author surname,
year, first distinctive word of the title, lowercase, no punctuation.

## Adding a paper

1. Drop the PDF in `literature/` named after the key you will use
2. Add the entry to `background.bib` with a DOI or PMID — that is what makes
   it retrievable by someone who does not have your copy
3. If it later gets cited, move the entry to `manuscript/references.bib`

`references.bib` stays exactly what the manuscript cites, so it can be audited
against the text. `background.bib` is the wider reading record and can hold
things that never get cited.

## Fetching what you do not have

Every entry carries a DOI. To get the PDF for an entry you lack:

    open "https://doi.org/<the doi from background.bib>"

For preprints, the arXiv or bioRxiv ID in the entry resolves the same way.
