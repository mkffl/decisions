

# Unit tests
Random seed set a the probability_monad level for repeatibility

`object RepeatableDistribution extends Distributions(new scala.util.Random(54321))`

scripts outputs scores and labels to feed into PYLLR and check that the results are the same

Script locations

scala: `scripts/prepUnitTests.sc`

python: `/Users/michel/Documents/pjs/model-discrimination-calibration/scripts/rocch_unit_test.py`

To run tests
```bash
./mill Decisions.test
```

# Running scripts
Load all code base in ammonite with 

```bash
./mill -i -w Decisions.repl
```

Note: had to upgrade to latest Mill version to avoid REPL errors.

mill also allows predef files to run scripts but these won't have the project modules in scope so had to copy paste.

# Optimisation (Apache Commons Math)
- Univariate functions http://commons.apache.org/proper/commons-math/javadocs/api-3.6/org/apache/commons/math3/optim/univariate/UnivariateObjectiveFunction.html
- Univariate Optimizer https://commons.apache.org/proper/commons-math/javadocs/api-3.6/org/apache/commons/math3/optim/univariate/UnivariateOptimizer.html#optimize(org.apache.commons.math3.optim.OptimizationData...)
- Goal Type https://commons.apache.org/proper/commons-math/javadocs/api-3.6/org/apache/commons/math3/optim/nonlinear/scalar/GoalType.html
- Brent Optimizer https://commons.apache.org/proper/commons-math/javadocs/api-3.6/org/apache/commons/math3/optim/univariate/BrentOptimizer.html#BrentOptimizer(double,%20double)
- Search Interval https://commons.apache.org/proper/commons-math/javadocs/api-3.6/org/apache/commons/math3/optim/univariate/SearchInterval.html
- Examples of Brent optimizer usage https://www.programcreek.com/java-api-examples/?api=org.apache.commons.math3.optim.univariate.BrentOptimizer
- Scipy equivalent https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize_scalar.html
