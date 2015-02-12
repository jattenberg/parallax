package com.dsi.parallax.optimization.stochastic;

import com.dsi.parallax.ml.vector.LinearVector;
import com.dsi.parallax.ml.vector.LinearVectorFactory;
import com.dsi.parallax.optimization.Gradient;
import com.dsi.parallax.optimization.Optimizable;
import com.dsi.parallax.optimization.regularization.GradientTruncation;
import com.dsi.parallax.optimization.regularization.LinearCoefficientLossType;
import com.dsi.parallax.optimization.stochastic.anneal.AnnealingSchedule;

import java.util.Map;
import java.util.Set;

import static com.google.common.base.Preconditions.checkArgument;

/**
 * Created by jattenberg on 2/10/15.
 */
public class ResilientPropagation extends AbstractGradientStochasticOptimizer {

    protected Gradient lastGrad = null;
    protected double stepGrow;
    protected double stepShrink;
    protected double minStep;
    protected double maxStep;

    public ResilientPropagation(int dimension, boolean bias, AnnealingSchedule annealingSchedule,
                                GradientTruncation truncation, Map<LinearCoefficientLossType, Double> coefficientWeights,
                                boolean regularizeIntercept, double regularizationWeight,
                                double stepGrow, double stepShrink, double minStep, double maxStep) {
        super(dimension, bias, annealingSchedule, truncation, coefficientWeights, regularizeIntercept, regularizationWeight);

        checkArgument(stepGrow > 1 && stepGrow > stepShrink,
                      "stepGrow must be in (1, inf), gt stepShrink, given %s", stepGrow);
        checkArgument(stepShrink > 0 && stepShrink < 1 && stepGrow > stepShrink,
                      "stepShrink must be in (0, 1), lt stepGrow, given %s", stepShrink);
        checkArgument(minStep > 0 && minStep < maxStep ,
                      "minStep must be in (0, inf), lt maxStep, given %s", minStep);
        checkArgument(maxStep > 0  && maxStep > minStep,
                      "maxStep must be in (0, inf), gt minStep, given %s", maxStep);

        this.stepGrow = stepGrow;
        this.stepShrink = stepShrink;
        this.maxStep = maxStep;
        this.minStep = minStep;

        lastGrad = new Gradient(LinearVectorFactory.getVector(dimension));
    }

    @Override
    protected Optimizable updateModel(Optimizable function, LinearVector regularizationGradient) {
        Gradient gradient = function.computeGradient().copy();
        gradient.plusEquals(regularizationGradient);

        return rPropUpdate(function, gradient);
    }

    @Override
    protected Optimizable updateModelWithRegularization(Optimizable function, LinearVector regularizationGradient) {
        return rPropUpdate(function, new Gradient(regularizationGradient));
    }

    protected Optimizable rPropUpdate(Optimizable function, Gradient gradient) {
        LinearVector parameter = function.getVector();

        checkArgument(
                parameter.size() == gradient.size(),
                "parameter (size %s) and gradient (size %s) should be of the same dimension",
                parameter.size(), gradient.size());

        LinearVector gCurrent = gradient.getGradientVector();
        LinearVector gLast = lastGrad.getGradientVector();

        Set<Integer> indices = gCurrent.getFeatureIndicies();
        for (int x_i : indices) {
            double g_prod = gCurrent.getValue(x_i)*gLast.getValue(x_i);
            double step = annealingSchedule.learningRate(epoch, x_i);

            if (g_prod > 0) {
                step *= stepGrow;
            } else if (g_prod < 0) {
                step *= stepShrink;
            }

            step = Math.min(step, maxStep);
            step = Math.max(step, minStep);

            double delta = -step*Math.signum(gCurrent.getValue(x_i));
            parameter.updateValue(x_i, delta);
        }

        function.setParameters(parameter);

        lastGrad = gradient.copy();
        return function;

    }
}
