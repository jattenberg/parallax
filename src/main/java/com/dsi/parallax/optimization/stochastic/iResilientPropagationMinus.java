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
 * Created by jattenberg on 2/12/15.
 */
public class iResilientPropagationMinus extends ResilientPropagation {

    protected LinearVector lastSteps;

    public iResilientPropagationMinus(int dimension, boolean bias, AnnealingSchedule annealingSchedule, GradientTruncation truncation, Map<LinearCoefficientLossType, Double> coefficientWeights, boolean regularizeIntercept, double regularizationWeight, double stepGrow, double stepShrink, double minStep, double maxStep) {
        super(dimension, bias, annealingSchedule, truncation, coefficientWeights, regularizeIntercept, regularizationWeight, stepGrow, stepShrink, minStep, maxStep);

        lastSteps = LinearVectorFactory.getVector(dimension);
        for (int x_i = 0; x_i < dimension; x_i++) {
            lastSteps.resetValue(x_i, annealingSchedule.learningRate(epoch, x_i));
        }
    }

    @Override
    protected Optimizable rPropUpdate(Optimizable function, Gradient gradient) {
        LinearVector parameter = function.getVector();

        checkArgument(
                parameter.size() == gradient.size(),
                "parameter (size %s) and gradient (size %s) should be of the same dimension",
                parameter.size(), gradient.size());

        LinearVector gCurrent = gradient.getGradientVector().copy();
        LinearVector gLast = lastGrad.getGradientVector();

        Set<Integer> indices = gCurrent.getFeatureIndicies();
        for (int x_i : indices) {
            double g_prod = gCurrent.getValue(x_i)*gLast.getValue(x_i);
            double step = lastSteps.getValue(x_i);

            if (g_prod > 0) {
                step *= stepGrow;
                step = Math.min(step, maxStep);
            } else if (g_prod < 0) {
                step *= stepShrink;
                step = Math.max(step, minStep);
                gCurrent.resetValue(x_i, 0.);
            }

            double delta = -step*Math.signum(gCurrent.getValue(x_i));
            parameter.updateValue(x_i, delta);
            lastSteps.resetValue(x_i, step);
        }

        function.setParameters(parameter);

        lastGrad = gradient;
        return function;

    }
}
