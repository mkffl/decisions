# Decisions

This code supports a series of blog articles about optimal decision-making, available at https://mkffl.github.io/2021/10/18/Decisions-Part-1.html

![Alt text](assets/blog-screenshot.png?raw=true "Screenshot")

With [mill](https://github.com/com-lihaoyi/mill) installed, you can run the script with 

```bash
./mill Decisions
```

and this will reproduce the blog analyses listed in the `main` method from `Decisions/src/Recipes.scala`.

### Organisation

This repository is only meant to be a learning aid - use at your own risk.

The `Tradeoff` class supports uncalibrated decision-making using common approaches based on the Bayes decision rule or ROC analysis (though the two are linked, see [part 2](https://mkffl.github.io/2021/10/28/Decisions-Part-2.html) ;) ). 

`Tradeoff` starts from raw scores and the ground truth, then applies transforms to get basic probability estimates like class-dependent cumulative distribution functions, which enable evaluations. Having the same interface for multiple evaluations helps me see the connections between them, which is my main motivation for starting from first principles.

Classes that inherit from `ErrorEstimator` support calibrated decision-making and closely follows [PYLLR](https://github.com/bsxfan/PYLLR). It would be interesting to rewrite it using a more scala-esque approach.

### Notes to self

#### Unit tests
To run tests
```bash
./mill Decisions.test
```

Random seed set a the probability_monad level for repeatibility:

`object RepeatableDistribution extends Distributions(new scala.util.Random(54321))`

APE tests are located in `ErrorEstimator`, which confirms that outputs ber outputs are similar to outputs from PYLLR.

`ammonite-scripts/prepUnitTests.sc`


#### Run

Load all code base in ammonite with 

```bash
./mill -i -w Decisions.repl
```

Note: had to upgrade to latest Mill version to avoid REPL errors.

mill also allows predef files to run scripts but these won't have the project modules in scope so had to copy paste.

#### Linting

```bash
./mill Decisions.reformat
```

#### Optimisation (Apache Commons Math)
- Univariate functions http://commons.apache.org/proper/commons-math/javadocs/api-3.6/org/apache/commons/math3/optim/univariate/UnivariateObjectiveFunction.html
- Univariate Optimizer https://commons.apache.org/proper/commons-math/javadocs/api-3.6/org/apache/commons/math3/optim/univariate/UnivariateOptimizer.html#optimize(org.apache.commons.math3.optim.OptimizationData...)
- Goal Type https://commons.apache.org/proper/commons-math/javadocs/api-3.6/org/apache/commons/math3/optim/nonlinear/scalar/GoalType.html
- Brent Optimizer https://commons.apache.org/proper/commons-math/javadocs/api-3.6/org/apache/commons/math3/optim/univariate/BrentOptimizer.html#BrentOptimizer(double,%20double)
- Search Interval https://commons.apache.org/proper/commons-math/javadocs/api-3.6/org/apache/commons/math3/optim/univariate/SearchInterval.html
- Examples of Brent optimizer usage https://www.programcreek.com/java-api-examples/?api=org.apache.commons.math3.optim.univariate.BrentOptimizer
- Scipy equivalent https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize_scalar.html

#### Plotly html
- By default Plotly.plot export generates a cdn in the "script" tags at the top of the file, which creates a layout issue in jekyll. A Mill task parses the html to replace the cdn with a simple string. 
- This simple replacement sovles the problem in Jekyll, where the plotly output is called as an embedded HTML using "include". The plotly cdn is now referenced in the jekyll parent layout. When the cdn is defined in the plotly output, it overrides the jekyll parent layout.
- I think a cleaner approach would generate a plotly html without the cdn, as explained [here](https://stackoverflow.com/questions/36262748/python-save-plotly-plot-to-local-file-and-insert-into-html). I tried saving with `useCdn=false` (see [docs](https://github.com/alexarchambault/plotly-scala/blob/288a31898914e36ab537b713d7acddd4b30ce59b/render/jvm/src/main/scala/plotly/Plotly.scala)) but that didn't work. There must be a better way, however, I don't have the patience/interest to dig further

