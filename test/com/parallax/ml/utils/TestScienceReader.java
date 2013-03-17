package com.parallax.ml.utils;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import com.parallax.ml.instance.BinaryClassificationInstance;
import com.parallax.ml.instance.BinaryClassificationInstances;

public class TestScienceReader {

	@Test
	public void testScienceReader() {
		BinaryClassificationInstances insts = ScienceReader.readScience();
		for (BinaryClassificationInstance inst : insts) {
			assertEquals(inst.size(), ScienceReader.DIMENSION);
			assertTrue(inst.getLabel() != null);
			for (int x_i : inst) {
				assertTrue(inst.getFeatureValue(x_i) > 0);
			}
		}
		assertTrue(insts.getDimensions() == ScienceReader.DIMENSION);
	}

}
