package com.dsi.parallax.optimization.stochastic;

import com.dsi.parallax.ml.util.bounds.GreaterThanValueBound;
import com.dsi.parallax.ml.util.bounds.LessThanOrEqualsValueBound;
import com.dsi.parallax.ml.util.bounds.LessThanValueBound;
import com.dsi.parallax.ml.util.option.Configuration;
import com.dsi.parallax.ml.util.option.FloatOption;

import static com.google.common.base.Preconditions.checkArgument;

/**
 * Created by jattenberg on 2/12/15.
 */
public class iResilientPropagationMinusBuilder extends StochasticGradientOptimizationBuilder<iResilientPropagationMinusBuilder> {


    private double maxStep = 1.;
    private double minStep = 0.000001;
    private double stepGrow = 1.2;
    private double stepShrink = 0.5;

    public iResilientPropagationMinusBuilder(int dimension, boolean bias) { super(dimension, bias); }

    public iResilientPropagationMinusBuilder setMaxStep(double maxStep) {
        checkArgument(maxStep > 0,"maxStep must be in (0, inf), given %s", maxStep);
        this.maxStep = maxStep;
        return this;
    }

    public iResilientPropagationMinusBuilder setMinStep(double minStep) {
        checkArgument(minStep > 0,"minStep must be in (0, inf), given %s", minStep);
        this.minStep = minStep;
        return this;
    }

    public iResilientPropagationMinusBuilder setStepGrow(double stepGrow) {
        checkArgument( stepGrow > 1, "stepGrow must be in (1, inf), given %s", stepGrow);
        this.stepGrow = stepGrow;
        return this;
    }

    public iResilientPropagationMinusBuilder setStepShrink(double stepShrink) {
        checkArgument( stepShrink > 0, "stepShrink must be in > 0, given %s", stepShrink);
        this.stepShrink = stepShrink;
        return this;
    }

    @Override
    public iResilientPropagationMinus build() {
        return new iResilientPropagationMinus(dimension, bias,
                annealingScheduleConfigurableBuilder.build(),
                truncationBuilder.build(), buildCoefficientLossMap(),
                regularizeIntercept, regularizationWeight,
                stepGrow, stepShrink, minStep, maxStep);

    }

    @Override
    protected iResilientPropagationMinusBuilder getThis() { return this; }

    @Override
    public void configure(Configuration<iResilientPropagationMinusBuilder> conf) {
        super.configure(conf);
        setMinStep(conf.floatOptionFromShortName("m"));
        setMaxStep(conf.floatOptionFromShortName("M"));
        setStepShrink(conf.floatOptionFromShortName("s"));
        setStepGrow(conf.floatOptionFromShortName("S"));
    }

    @Override
    public Configuration<iResilientPropagationMinusBuilder> getConfiguration() {
        Configuration<iResilientPropagationMinusBuilder> conf = new Configuration<iResilientPropagationMinusBuilder>(
                new iResilientPropagationMinusBuilderOptions());
        populateConfiguration(conf);
        return conf;
    }

    @Override
    public Configuration<iResilientPropagationMinusBuilder> populateConfiguration(
            Configuration<iResilientPropagationMinusBuilder> conf) {
        super.populateConfiguration(conf);
        conf.addFloatValueOnShortName("s", stepShrink);
        conf.addFloatValueOnShortName("S", stepGrow);
        conf.addFloatValueOnShortName("m", minStep);
        conf.addFloatValueOnShortName("M", maxStep);
        return conf;
    }

    public static iResilientPropagationMinusBuilderOptions getOptions() {
        return new iResilientPropagationMinusBuilderOptions();
    }

    public static class iResilientPropagationMinusBuilderOptions extends
            StochasticGradientOptimizationBuilder.StochasticGradientOptimizationBuilderOptions<iResilientPropagationMinusBuilder> {
        {
            addOption(new FloatOption("m", "minstep",
                    "minimum possible step size",
                    0.000001, false, new GreaterThanValueBound(0),
                    new LessThanValueBound(BIGVAL)));
            addOption(new FloatOption("M", "maxstep",
                    "minimum possible step size",
                    1, false, new GreaterThanValueBound(0),
                    new LessThanOrEqualsValueBound(BIGVAL)));
            addOption(new FloatOption("s", "stepshrink",
                    "shift if gradient signs differ",
                    0.5, false, new GreaterThanValueBound(0),
                    new LessThanValueBound(1)));
            addOption(new FloatOption("S", "stepgrow",
                    "shift if gradient signs are the same",
                    1.2, false, new GreaterThanValueBound(1),
                    new LessThanOrEqualsValueBound(BIGVAL)));
        }

    }
}