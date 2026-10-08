# Phase 10 manuscript

Build `phase10_last_two_capacities.tex` using `sh build.sh` in this directory. The script uses three pdfLaTeX passes with shell escape disabled and copies the result over the bundled PDF; use a disposable copy to preserve the original PDF bytes.

The inherited Phase 9 analytic upper estimate and endpoint lower proof are attributed in the manuscript. [BUILD_PROVENANCE.json](BUILD_PROVENANCE.json) gives their exact historical source hashes and the current public locations. The independent upper proof and both anchor variants are retained in `../research/`. The fixed-location variant has an N-dependent threshold; the final far-anchor construction has a threshold uniform in N.

This working paper is not external peer review or formal verification. See the stage status and technical review for its exact scope.
