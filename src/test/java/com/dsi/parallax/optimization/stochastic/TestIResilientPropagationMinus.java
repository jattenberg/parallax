package com.dsi.parallax.optimization.stochastic;

import com.dsi.parallax.optimization.utilities.OptimizationTestUtils;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

/**
 * Created by jattenberg on 2/12/15.
 */
public class TestIResilientPropagationMinus {

    @Test
    public void TestOptimizes() {
        iResilientPropagationMinusBuilder builder = new iResilientPropagationMinusBuilder(2, false);
        iResilientPropagationMinus sgd = builder.build();

        OptimizationTestUtils.FunctionOne f1 = new OptimizationTestUtils.FunctionOne();
        double distToMin = OptimizationTestUtils.distToMinimum(f1.getVector()
                .getW());

        for (int i = 0; i < 150; i++) {

            double loss = distToMin;
            sgd.update(f1);
            distToMin = f1.computeLoss();
        }

        assertEquals(0, distToMin, 0.001);
    }

    @Test
    public void TestOptimizes2() {
        iResilientPropagationMinusBuilder builder = new iResilientPropagationMinusBuilder(1, false);
        iResilientPropagationMinus sgd = builder.build();

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
                0.00001);
    }
}
