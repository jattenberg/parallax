package com.dsi.parallax.optimization.stochastic;

import com.dsi.parallax.ml.util.bounds.GreaterThanValueBound;
import com.dsi.parallax.ml.util.bounds.LessThanOrEqualsValueBound;
import com.dsi.parallax.ml.util.bounds.LessThanValueBound;
import com.dsi.parallax.ml.util.option.Configuration;
import com.dsi.parallax.ml.util.option.FloatOption;

import static com.google.common.base.Preconditions.checkArgument;

/**
 * Created by jattenberg on 2/4/15.
 */
public class ADAMBuilder extends StochasticGradientOptimizationBuilder<ADAMBuilder> {

    private double b1 = 0.1, b2 = 0.001, lambda = 0.00000001, alpha = 0.0002, epsilon = 0.00000001;

    public ADAMBuilder(int dimension, boolean bias) { super(dimension, bias); }

    public ADAMBuilder setB1(double b1) {
        checkArgument( b1 > 0 && b1 <= 1, "b1 must be in (0, 1], given %s", b1);
        this.b1 = b1;
        return this;
    }

    public ADAMBuilder setB2(double b2) {
        checkArgument( b2 > 0 && b2 <= 1, "b2 must be in (0, 1], given %s", b2);
        this.b2 = b2;
        return this;
    }

    public ADAMBuilder setLambda(double lambda) {
        checkArgument( lambda > 0 && lambda < 1, "lambda must be in (0, 1), given %s", lambda);
        this.lambda = lambda;
        return this;
    }

    public ADAMBuilder setAlpha(double alpha) {
        checkArgument( alpha > 0, "alpha must be in > 0, given %s", alpha);
        this.alpha = alpha;
        return this;
    }

    public ADAMBuilder setEpsilon(double epsilon) {
        checkArgument( epsilon > 0, "epsilon must be in > 0, given %s", epsilon);
        this.epsilon = epsilon;
        return this;
    }

    @Override
    public ADAM build() {
        return new ADAM(dimension, bias,
                annealingScheduleConfigurableBuilder.build(),
                truncationBuilder.build(), buildCoefficientLossMap(),
                regularizeIntercept, regularizationWeight,
                b1, b2, lambda, alpha, epsilon);

    }

    @Override
    protected ADAMBuilder getThis() { return this; }

    @Override
    public void configure(Configuration<ADAMBuilder> conf) {
        super.configure(conf);
        setB1(conf.floatOptionFromShortName("b1"));
        setB2(conf.floatOptionFromShortName("b2"));
        setLambda(conf.floatOptionFromShortName("l"));
        setEpsilon(conf.floatOptionFromShortName("e"));
        setAlpha(conf.floatOptionFromShortName("a"));
    }

    @Override
    public Configuration<ADAMBuilder> getConfiguration() {
        Configuration<ADAMBuilder> conf = new Configuration<ADAMBuilder>(
                new ADAMBuilderOptions());
        populateConfiguration(conf);
        return conf;
    }

    @Override
    public Configuration<ADAMBuilder> populateConfiguration(
            Configuration<ADAMBuilder> conf) {
        super.populateConfiguration(conf);
        conf.addFloatValueOnShortName("b1", b1);
        conf.addFloatValueOnShortName("b2", b2);
        conf.addFloatValueOnShortName("a", alpha);
        conf.addFloatValueOnShortName("l", lambda);
        conf.addFloatValueOnShortName("e", epsilon);
        return conf;
    }

    public static ADAMBuilderOptions getOptions() {
        return new ADAMBuilderOptions();
    }

    public static class ADAMBuilderOptions extends
            StochasticGradientOptimizationBuilderOptions<ADAMBuilder> {
        {
            addOption(new FloatOption("l", "lambda",
                    "lambda in ADAM model",
                    0.00000001, false, new GreaterThanValueBound(0),
                    new LessThanValueBound(1)));
            addOption(new FloatOption("a", "alpha",
                    "alpha in ADAM model, controlling step size",
                    0.0002, false, new GreaterThanValueBound(0),
                    new LessThanOrEqualsValueBound(BIGVAL)));
            addOption(new FloatOption("e", "epsilon",
                    "epsilon in ADAM model",
                    0.00000001, false, new GreaterThanValueBound(0),
                    new LessThanOrEqualsValueBound(BIGVAL)));
            addOption(new FloatOption("b1", "b1",
                    "b1 in ADAM model",
                    0.1, false, new GreaterThanValueBound(0),
                    new LessThanOrEqualsValueBound(BIGVAL)));
            addOption(new FloatOption("b2", "b2",
                    "b2 in ADAM model",
                    0.001, false, new GreaterThanValueBound(0),
                    new LessThanOrEqualsValueBound(BIGVAL)));
        }

    }
}
