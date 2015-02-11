package com.dsi.parallax.optimization.stochastic;

import com.dsi.parallax.ml.util.MLUtils;
import com.dsi.parallax.optimization.utilities.OptimizationTestUtils;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Created by jattenberg on 2/10/15.
 */
public class TestResilientPropagation {


    @Test
    public void TestOptimizes() {
        ResilientPropagationBuilder builder = new ResilientPropagationBuilder(2, false)
                .setStepGrow(70);
        ResilientPropagation sgd = builder.build();

        OptimizationTestUtils.FunctionOne f1 = new OptimizationTestUtils.FunctionOne();
        double distToMin = OptimizationTestUtils.distToMinimum(f1.getVector()
                .getW());

        for (int i = 0; i < 1500; i++) {

            double loss = distToMin;
            sgd.update(f1);
            distToMin = f1.computeLoss();

        }

        assertEquals(0, distToMin, 0.001);
    }

    @Test
    public void TestOptimizes2() {
        ResilientPropagationBuilder builder = new ResilientPropagationBuilder(1, false)
                .setStepGrow(70);
        ResilientPropagation sgd = builder.build();

        OptimizationTestUtils.FunctionTwo f2 = new OptimizationTestUtils.FunctionTwo();
        double distToMin = OptimizationTestUtils.distToMinimum2(f2.getVector()
                .getW());
        for (int i = 0; i < 200; i++) {
            double loss = distToMin;
            sgd.update(f2);
            distToMin = OptimizationTestUtils.distToMinimum2(f2.getVector()
                    .getW());
        }
        assertEquals(f2.getParameter(0), OptimizationTestUtils.minimum2[0],
                0.0000001);
    }
}
