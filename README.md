# Unit tests
Random seed set a the probability_monad level for repeatibility

`object RepeatableDistribution extends Distributions(new scala.util.Random(54321))`

scripts outputs scores and labels to feed into PYLLR and check that the results are the same

Script locations

scala: `scripts/prepUnitTests.sc`

python: `/Users/michel/Documents/pjs/model-discrimination-calibration/scripts/rocch_unit_test.py`


# Running scripts
Load all code base in ammonite with 

```bash
./mill -i -w Decisions.repl
```

Note: had to upgrade to latest Mill version to avoid REPL errors.

mill also allows predef files to run scripts but these won't have the project modules in scope so had to copy paste.