package com.dsi.parallax.ml.evaluation;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import com.dsi.parallax.ml.evaluation.ReceiverOperatingCharacteristic;

public class TestReceiverOperatingCharacteristic {

	static double[] labels = { 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0 };
	static double[] predictions = { 0.80962, 0.48458, 0.65812, 0.16117,
			0.47375, 0.26587, 0.71517, 0.63866, 0.36296, 0.89639, 0.35936,
			0.22413, 0.36402, 0.41459, 0.83148, 0.23271 };
	static int size = 16;

	@Test
	public void testAUC() {
		ReceiverOperatingCharacteristic roc = new ReceiverOperatingCharacteristic();
		for (int i = 0; i < 16; i++) {
			roc.add(labels[i], predictions[i]);
		}
		// test value comes from scikits learn
		assertEquals(0.96825396825396826, roc.binaryAUC(), 0.000000001);
	}

	@Test
	public void testCurve() {
		double[][] ROCs = { { 0.0, 0.0 }, { 0.0, 0.1111111111111111 },
				{ 0.0, 0.2222222222222222 }, { 0.0, 0.3333333333333333 },
				{ 0.0, 0.4444444444444444 }, { 0.0, 0.5555555555555556 },
				{ 0.0, 0.6666666666666666 }, { 0.0, 0.7777777777777778 },
				{ 0.14285714285714285, 0.7777777777777778 },
				{ 0.14285714285714285, 0.8888888888888888 },
				{ 0.14285714285714285, 1.0 }, { 0.2857142857142857, 1.0 },
				{ 0.42857142857142855, 1.0 }, { 0.5714285714285714, 1.0 },
				{ 0.7142857142857143, 1.0 }, { 0.8571428571428571, 1.0 },
				{ 1.0, 1.0 } };
		ReceiverOperatingCharacteristic roc = new ReceiverOperatingCharacteristic();
		for (int i = 0; i < 16; i++) {
			roc.add(labels[i], predictions[i]);
		}

		double[][] curve = roc.ROC();
		assertEquals(ROCs.length, curve.length);

		for (int i = 0; i < curve.length; i++) {
			double[] point = curve[i];
			double[] truePoint = ROCs[i];
			assertEquals(truePoint.length, point.length);
			for (int j = 0; j < truePoint.length; j++) {
				assertEquals(truePoint[j], point[j], 0.000000001);
			}
		}
	}

}
