package com.dsi.parallax.optimization.stochastic;

import com.dsi.parallax.ml.vector.LinearVector;
import com.dsi.parallax.ml.vector.LinearVectorFactory;
import com.dsi.parallax.optimization.Gradient;
import com.dsi.parallax.optimization.Optimizable;
import com.dsi.parallax.optimization.regularization.GradientTruncation;
import com.dsi.parallax.optimization.regularization.LinearCoefficientLossType;
import com.dsi.parallax.optimization.stochastic.anneal.AnnealingSchedule;
import java.util.Set;

import java.util.Map;

import static com.google.common.base.Preconditions.checkArgument;

/**c
 * Created by jattenberg on 1/31/15.
 * ADAM: A METHOD FOR STOCHASTIC OPTIMIZATION
 * http://arxiv.org/pdf/1412.6980v2.pdf
 */
public class ADAM extends AbstractGradientStochasticOptimizer{

    private LinearVector m, v;
    private double b1, b2, lambda, alpha, epsilon;
    private double b1t;

    protected ADAM(int dimension, boolean bias, AnnealingSchedule schedule, GradientTruncation truncation, Map<LinearCoefficientLossType, Double> coefficientWeights, boolean regularizeIntercept, double regularizationWeight, double b1, double b2, double lambda, double alpha, double epsilon) {
        super(dimension, bias, schedule, truncation, coefficientWeights, regularizeIntercept, regularizationWeight);

        checkArgument( b1 > 0 && b1 <= 1, "b1 must be in (0, 1], given %s", b1);
        checkArgument( b2 > 0 && b2 <= 1, "b2 must be in (0, 1], given %s", b2);
        checkArgument( lambda > 0 && lambda < 1, "lambda must be in (0, 1), given %s", lambda);
        checkArgument( alpha > 0, "alpha must be in > 0, given %s", alpha);
        checkArgument( epsilon > 0, "epsilon must be in > 0, given %s", epsilon);

        this.b1 = b1;
        this.b2 = b2;
        this.alpha = alpha;
        this.lambda = lambda;
        this.epsilon = epsilon;

        m = LinearVectorFactory.getVector(dimension);
        v = LinearVectorFactory.getVector(dimension);

    }

    @Override
    protected Optimizable updateModel(Optimizable function, LinearVector regularizationGradient) {
        Gradient gradient = function.computeGradient().copy();
        gradient.plusEquals(regularizationGradient);

        return ADAMUpdate(function, gradient);
    }

    @Override
    protected Optimizable updateModelWithRegularization(Optimizable function, LinearVector regularizationGradient) {
        return ADAMUpdate(function, new Gradient(regularizationGradient));
    }

    private Optimizable ADAMUpdate(Optimizable function, Gradient gradient) {
        LinearVector parameter = function.getVector();

        checkArgument(
                parameter.size() == gradient.size(),
                "parameter (size %s) and gradient (size %s) should be of the same dimension",
                parameter.size(), gradient.size());
        epoch++;
        b1t = 1.0 - (1.0 - b1)*lambda;
        m = m.times(1.0 - b1t).plusEqualsVectorTimes(gradient.getGradientVector(), b1t);
        v = v.times(1.0 - b2).plusEquals(gradient.copy().times(gradient).times(b2));

        Set<Integer> keys = v.getFeatureIndicies();
        keys.addAll(m.getFeatureIndicies());

        for (int x_i : keys) {
            double mhat = m.getValue(x_i)/(1.0 - Math.pow(1.0 - b1, epoch));
            double vhat = v.getValue(x_i)/(1.0 - Math.pow(1.0 - b2, epoch));
            double update = -alpha*mhat/(Math.sqrt(vhat) + epsilon);

            parameter.updateValue(x_i, update);
        }

        function.setParameters(parameter);
        return function;
    }
}
