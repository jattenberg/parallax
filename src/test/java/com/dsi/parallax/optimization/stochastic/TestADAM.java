package com.dsi.parallax.optimization.stochastic;

import com.dsi.parallax.ml.util.MLUtils;
import com.dsi.parallax.optimization.utilities.OptimizationTestUtils;
import org.junit.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Created by jattenberg on 2/4/15.
 */
public class TestADAM {


    @Test
    public void TestOptimizes() {
        ADAMBuilder builder = new ADAMBuilder(2, false).setAlpha(0.09);
        ADAM sgd = builder.build();

        OptimizationTestUtils.FunctionOne f1 = new OptimizationTestUtils.FunctionOne();
        double distToMin = OptimizationTestUtils.distToMinimum(f1.getVector()
                .getW());

        for (int i = 0; i < 150; i++) {

            double loss = distToMin;
            sgd.update(f1);
            distToMin = f1.computeLoss();
            assertTrue(distToMin < loss
                    || MLUtils.floatingPointEquals(distToMin, 0));
        }

        assertEquals(0, distToMin, 0.001);
    }

    @Test
    public void TestOptimizes2() {
        ADAMBuilder builder = new ADAMBuilder(1, false).setAlpha(0.1);
        ADAM sgd = builder.build();

        OptimizationTestUtils.FunctionTwo f2 = new OptimizationTestUtils.FunctionTwo();
        double distToMin = OptimizationTestUtils.distToMinimum2(f2.getVector()
                .getW());
        for (int i = 0; i < 25; i++) {
            double loss = distToMin;
            sgd.update(f2);
            distToMin = OptimizationTestUtils.distToMinimum2(f2.getVector()
                    .getW());
            assertTrue(distToMin < loss
                    || MLUtils.floatingPointEquals(distToMin, 0));
        }
        assertEquals(f2.getParameter(0), OptimizationTestUtils.minimum2[0],
                0.0000001);
    }
}
