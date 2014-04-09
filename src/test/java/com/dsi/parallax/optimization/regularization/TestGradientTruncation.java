/*******************************************************************************
 * Copyright 2012 Josh Attenberg. Not for re-use or redistribution.
 ******************************************************************************/
package com.dsi.parallax.optimization.regularization;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import com.dsi.parallax.ml.util.MLUtils;
import com.dsi.parallax.ml.vector.LinearVector;
import com.dsi.parallax.ml.vector.LinearVectorFactory;
import com.dsi.parallax.optimization.regularization.GradientTruncation;
import com.dsi.parallax.optimization.regularization.TruncationConfigurableBuilder;
import com.dsi.parallax.optimization.regularization.TruncationType;

public class TestGradientTruncation {

	@Test
	public void test() {
		for(TruncationType type : TruncationType.values()) {
			LinearVector vect = LinearVectorFactory.getVector(1000);
			for(int i = 0; i < 1000; i++)
				vect.resetValue(i, MLUtils.GENERATOR.nextDouble());
			TruncationConfigurableBuilder builder = new TruncationConfigurableBuilder();
			builder.setTruncationType(type).setAlpha(type == TruncationType.MODDUCHI ? 1050 : 600).setThreshold(.01).setPeriod(1);
			GradientTruncation trunc = builder.build();
			
			double beforeTwoNorm = vect.L2Norm();
			double beforeOneNorm = vect.L1Norm();
			
			trunc.truncateParameters(vect);
			
			double afterTwoNorm = vect.L2Norm();
			double afterOneNorm = vect.L1Norm();
			
			if(type.equals(TruncationType.NONE)) {
				assertEquals(beforeOneNorm, afterOneNorm, .00000001);
				assertEquals(beforeTwoNorm, afterTwoNorm, .00000001);
			} else {
				
				assertTrue(afterOneNorm < beforeOneNorm);
				assertTrue(afterTwoNorm < beforeTwoNorm);
			}
				
		}
	}

}
