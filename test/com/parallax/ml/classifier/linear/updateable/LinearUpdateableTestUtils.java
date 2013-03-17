package com.parallax.ml.classifier.linear.updateable;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import com.parallax.ml.classifier.linear.optimizable.AbstractGradientUpdateableClassifier;
import com.parallax.ml.classifier.linear.optimizable.GradientUpdateableClassifierConfigurableBuilder;
import com.parallax.ml.instance.BinaryClassificationInstances;
import com.parallax.ml.utils.IrisReader;
import com.parallax.optimization.regularization.TruncationConfigurableBuilder;
import com.parallax.optimization.regularization.TruncationType;

public class LinearUpdateableTestUtils {

	public static <C extends AbstractLinearUpdateableClassifier<C>, B extends LinearUpdateableClassifierBuilder<C, B>> void testRegularization(
			B builder) {
		BinaryClassificationInstances insts = IrisReader.readIris();
		builder.setDimension(IrisReader.DIMENSION).setBias(true);
		
		C baselineModel = builder.build();
		baselineModel.train(insts);

		double baselineL0 = baselineModel.getVector().L0Norm();
		double baselineL1 = baselineModel.getVector().L1Norm();
		double baselineL2 = baselineModel.getVector().L2Norm();
		
		assertTrue(baselineL0 > 0);
		assertTrue(baselineL1 > 0);
		assertTrue(baselineL2 > 0);
		//test to regularization

	}
	public static <C extends AbstractLinearUpdateableClassifier<C>, B extends LinearUpdateableClassifierBuilder<C, B>> void testTruncation(
			B builder) {
		BinaryClassificationInstances insts = IrisReader.readIris();
		builder.setDimension(IrisReader.DIMENSION).setBias(true);

		C baselineModel = builder.build();
		baselineModel.train(insts);

		double baselineL0 = baselineModel.getVector().L0Norm();
		double baselineL1 = baselineModel.getVector().L1Norm();
		double baselineL2 = baselineModel.getVector().L2Norm();

		// Test no truncation
		TruncationConfigurableBuilder truncBuilder = new TruncationConfigurableBuilder();
		truncBuilder.setTruncationType(TruncationType.NONE).setAlpha(5)
				.setPeriod(10).setThreshold(10);
		builder.setTruncationBuilder(truncBuilder);
		C model = builder.build();

		model.train(insts);
		double L0 = model.getVector().L0Norm();
		double L1 = model.getVector().L1Norm();
		double L2 = model.getVector().L2Norm();

		assertEquals(L0, baselineL0, 0.00001);
		assertEquals(L1, baselineL1, 0.00001);
		assertEquals(L2, baselineL2, 0.00001);

		for (TruncationType type : TruncationType.values()) {
			if (type == TruncationType.NONE) {
				continue;
			}
			
			// test no truncation
			truncBuilder.setTruncationType(type).setAlpha(0).setPeriod(5)
					.setThreshold(0);
			builder.setTruncationBuilder(truncBuilder);
			C untruncatedModel = builder.build();

			untruncatedModel.train(insts);
			double untruncatedL0 = untruncatedModel.getVector().L0Norm();
			double untruncatedL1 = untruncatedModel.getVector().L1Norm();
			double untruncatedL2 = untruncatedModel.getVector().L2Norm();

			assertEquals(untruncatedL0, baselineL0, 0.00001);
			assertEquals(untruncatedL1, baselineL1, 0.00001);
			assertEquals(untruncatedL2, baselineL2, 0.00001);

			
			
			// test some truncation
			// different parameters have different effects depending
			// on the type of truncation used. 
			truncBuilder
					.setTruncationType(type)
					.setAlpha(
							type == TruncationType.MODDUCHI
									|| type == TruncationType.PEGASOS ? 25d
									: .5)
					.setPeriod(5)
					.setThreshold(
							type == TruncationType.ROUNDING
									|| type == TruncationType.TRUNCATING ? 4
									: 1);
			builder.setTruncationBuilder(truncBuilder);
			C truncatedModel = builder.build();

			truncatedModel.train(insts);
			double truncatedL0 = truncatedModel.getVector().L0Norm();
			double truncatedL1 = truncatedModel.getVector().L1Norm();
			double truncatedL2 = truncatedModel.getVector().L2Norm();

			assertTrue(truncatedL0 <= baselineL0);
			assertTrue(truncatedL1 < baselineL1);
			assertTrue(truncatedL2 < baselineL2);

			// test complete truncation
			truncBuilder.setTruncationType(type).setAlpha(10000).setPeriod(1)
					.setThreshold(10000);
			builder.setTruncationBuilder(truncBuilder);
			C fullyTruncatedModel = builder.build();

			fullyTruncatedModel.train(insts);
			double fullyTruncatedL0 = fullyTruncatedModel.getVector().L0Norm();
			double fullyTruncatedL1 = fullyTruncatedModel.getVector().L1Norm();
			double fullyTruncatedL2 = fullyTruncatedModel.getVector().L2Norm();

			if (type == TruncationType.PEGASOS) {
				assertTrue(truncatedL0 >= fullyTruncatedL0);
				assertTrue(truncatedL1 > fullyTruncatedL1);
				assertTrue(truncatedL2 > fullyTruncatedL2);

			} else {
				assertEquals(fullyTruncatedL0, 0, 0.00001);
				assertEquals(fullyTruncatedL1, 0, 0.00001);
				assertEquals(fullyTruncatedL2, 0, 0.00001);
			}
		}
	}

	public static <C extends AbstractGradientUpdateableClassifier<C>, B extends GradientUpdateableClassifierConfigurableBuilder<C, B>> void testTruncationOptimizable(
			B builder) {
		BinaryClassificationInstances insts = IrisReader.readIris();
		builder.setDimension(IrisReader.DIMENSION).setBias(true);

		C baselineModel = builder.build();
		baselineModel.train(insts);

		double baselineL0 = baselineModel.getVector().L0Norm();
		double baselineL1 = baselineModel.getVector().L1Norm();
		double baselineL2 = baselineModel.getVector().L2Norm();

		// Test no truncation
		TruncationConfigurableBuilder truncBuilder = new TruncationConfigurableBuilder();
		truncBuilder.setTruncationType(TruncationType.NONE).setAlpha(5)
				.setPeriod(10).setThreshold(10);
		builder.setTruncationBuilder(truncBuilder);
		C model = builder.build();

		model.train(insts);
		double L0 = model.getVector().L0Norm();
		double L1 = model.getVector().L1Norm();
		double L2 = model.getVector().L2Norm();

		assertEquals(L0, baselineL0, 0.00001);
		assertEquals(L1, baselineL1, 0.00001);
		assertEquals(L2, baselineL2, 0.00001);

		for (TruncationType type : TruncationType.values()) {
			if (type == TruncationType.NONE) {
				continue;
			}

			// test no truncation
			truncBuilder.setTruncationType(type).setAlpha(0).setPeriod(5)
					.setThreshold(0);
			builder.setTruncationBuilder(truncBuilder);
			C untruncatedModel = builder.build();

			untruncatedModel.train(insts);
			double untruncatedL0 = untruncatedModel.getVector().L0Norm();
			double untruncatedL1 = untruncatedModel.getVector().L1Norm();
			double untruncatedL2 = untruncatedModel.getVector().L2Norm();

			assertEquals(untruncatedL0, baselineL0, 0.00001);
			assertEquals(untruncatedL1, baselineL1, 0.00001);
			assertEquals(untruncatedL2, baselineL2, 0.00001);

			// test some truncation
			truncBuilder
					.setTruncationType(type)
					.setAlpha(
							type == TruncationType.MODDUCHI
									|| type == TruncationType.PEGASOS ? 45d
									: .5)
					.setPeriod(5)
					.setThreshold(
							type == TruncationType.ROUNDING
									|| type == TruncationType.TRUNCATING ? 20
									: 1);
			builder.setTruncationBuilder(truncBuilder);
			C truncatedModel = builder.build();

			truncatedModel.train(insts);
			double truncatedL0 = truncatedModel.getVector().L0Norm();
			double truncatedL1 = truncatedModel.getVector().L1Norm();
			double truncatedL2 = truncatedModel.getVector().L2Norm();
			
			assertTrue(truncatedL0 <= baselineL0);
			if (type == TruncationType.PEGASOS) {
				assertTrue(truncatedL1 <= baselineL1);
				assertTrue(truncatedL2 <= baselineL2);
				
			} else {
				assertTrue(truncatedL1 < baselineL1);
				assertTrue(truncatedL2 < baselineL2);
			}

			// test complete truncation
			truncBuilder.setTruncationType(type).setAlpha(10000).setPeriod(1)
					.setThreshold(10000);
			builder.setTruncationBuilder(truncBuilder);
			C fullyTruncatedModel = builder.build();

			fullyTruncatedModel.train(insts);
			double fullyTruncatedL0 = fullyTruncatedModel.getVector().L0Norm();
			double fullyTruncatedL1 = fullyTruncatedModel.getVector().L1Norm();
			double fullyTruncatedL2 = fullyTruncatedModel.getVector().L2Norm();

			if (type == TruncationType.PEGASOS) {
				assertTrue(truncatedL0 >= fullyTruncatedL0);
				assertTrue(truncatedL1 > fullyTruncatedL1);
				assertTrue(truncatedL2 > fullyTruncatedL2);

			} else {
				assertEquals(fullyTruncatedL0, 0, 0.00001);
				assertEquals(fullyTruncatedL1, 0, 0.00001);
				assertEquals(fullyTruncatedL2, 0, 0.00001);
			}
		}
	}

}
