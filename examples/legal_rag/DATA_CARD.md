# Data card: fictional legal-like fixture

`fixtures/sources.json` holds six source records in five families, schema
`minirag-legal-fixture` version 1. The legal RAG example will be built on it.
It exists to exercise software paths: versions, conflicts, jurisdictions,
tenant isolation and unknown metadata. **It is not law, describes no real legal
system, and nothing built on it is legal advice.**

## Provenance

- **Origin.** All text is original and fictional. It was drafted with an AI
  coding assistant for this repository on 2026-11-02. It is not adapted from
  any statute, regulation, judgment, contract, client file or other private
  material.
- **Names.** Varnholt, Quillmouth, Orchidwake Shipping and Basaltine Marine are
  invented, as are the "credits" currency and every date, number and page.
  Any match with a real place, body or company is coincidental. Jurisdiction
  codes start with `fic-` so a real code cannot pass validation.
- **Licence.** MIT, the same as the repository. Every record carries
  `"license": "MIT"` and `"origin": "original-fictional"`. Every title ends
  with "(fictional)", and every text opens with the line `FICTIONAL TEST TEXT.
  Not law and not legal advice.` The validator enforces all four.

## The sources

| `source_id` | Scope (tenant / matter) | Jurisdiction / regime | Authority | Effective `[from, to)` | Labels | Why it is here |
| --- | --- | --- | --- | --- | --- | --- |
| `fic-harbour-safety-act@v1` | public | `fic-varnholt` / `maritime-safety` | statute | 2019-01-01 to 2023-07-01 | superseded-version | Hull inspection every 24 months. Replaced by v2. |
| `fic-harbour-safety-act@v2` | public | `fic-varnholt` / `maritime-safety` | statute | 2023-07-01, no end | current-version | Every 12 months. Supersedes v1 on the day v1 ends. |
| `fic-harbour-inspection-guidance@v1` | public | `fic-varnholt` / `maritime-safety` | guidance | 2023-09-01, no end | conflicting-source | 18 months for 20–30 m vessels, which contradicts v2. It does not say which one prevails. |
| `fic-quillmouth-port-code@v1` | public | `fic-quillmouth` / `maritime-safety` | regulation | 2021-03-15, end unknown | other-jurisdiction, unknown-metadata | Every 6 months, but in another jurisdiction. Pages unknown. |
| `fic-orchid-fleet-memo@v1` | `tenant-orchid` / `matter-orchid-001` | `fic-varnholt` / `maritime-safety` | internal-memo | 2024-02-10, no end | tenant-private | A client document only one matter may see. |
| `fic-basalt-file-note@v1` | `tenant-basalt` / tenant-wide | unknown / unknown | unknown | unknown to unknown | tenant-private, unknown-metadata | Every field that may be unknown is unknown. |

The labels say why a record exists. They are not answers: questions and
expected evidence for these sources come in a separate, frozen evaluation set,
which will record who wrote its labels and how.

## What a record guarantees

[`provenance.py`](provenance.py) defines the contract, and its module docstring
gives every rule. In short:

- **Every key is required.** A fact that may be unknown is written `"unknown"`.
  Null means "none": no matter, no end date, supersedes nothing. Look-alikes
  such as `"Unknown"`, `"n/a"`, `""` and `"tenant-unknown"` are rejected.
- **Scope is never unknown.** `tenant` is `public` or `tenant-...`, and a
  public record has no matter. A matter belongs to one tenant. A record nobody
  can be shown to own does not load at all.
- **A family agrees with itself.** Every version of a family has the same
  tenant, matter, jurisdiction, regime and authority type.
- **Dates and spans are half-open.** A version is in force on
  `effective_from` and not on `effective_to`. Section `start`/`end` are
  code-point offsets into `text` (Python string indices, not bytes). Each
  section starts at the beginning of a line, and its first line is exactly its
  heading. Pages are 1-based and inclusive.
- **Versions form one chain.** Each later version supersedes the one just
  before it. A version that is followed needs an end date or `"unknown"`, never
  null. Where the dates compared are known, versions hand over on one day,
  never overlap, and none ends before an earlier one starts.
- **`content_hash`** is the SHA-256 of the NFC-normalised UTF-8 text. It catches
  accidental edits and drift. Anyone can recompute it, so it does not prove
  who wrote the text.

Check the bundled file, or your own, from the repository root:

```bash
.venv/bin/python -m examples.legal_rag.provenance                  # the bundled fixture
.venv/bin/python -m examples.legal_rag.provenance my-sources.json  # exit 1 lists the problems found
```

```
6 valid records: 5 families, 3 tenant/matter scopes
```

## Limitations

- **Six records exercise code paths. They cannot measure anything.** No
  retrieval-quality, citation or legal-accuracy claim can rest on them.
- **Validation is structural.** It proves a record is complete and internally
  consistent. It cannot prove a text is fictional, that a tenant or matter was
  assigned correctly, or that a version history matches any real legislation.
- **Unknown dates are not checked for overlap.** If a version's start or end
  is `"unknown"`, the validator cannot tell whether it overlaps its neighbours.
  A filter by date must treat it as unknown, not as a match.
- **No precedence is encoded.** The fixture does not say whether a statute
  beats guidance, and nothing built on it should infer that.
- **The layout is minimal.** Plain English text: no tables, footnotes,
  schedules, amendments inside a text, or cross-references beyond the words of
  the text itself. Page numbers only demonstrate the page-span fields.
- **The hash is not a signature.** See above.
