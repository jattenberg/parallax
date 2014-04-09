package com.dsi.parallax.ml.utils;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import com.dsi.parallax.ml.instance.BinaryClassificationInstance;
import com.dsi.parallax.ml.instance.BinaryClassificationInstances;
import com.dsi.parallax.ml.utils.IrisReader;

public class TestIrisReader {

	@Test
	public void testIrisReader() {
		BinaryClassificationInstances insts = IrisReader.readIris();
		for (BinaryClassificationInstance inst : insts) {
			assertEquals(inst.size(), 4);
			assertTrue(inst.getLabel() != null);
			for (int x_i : inst) {
				assertTrue(inst.getFeatureValue(x_i) > 0);
			}
		}
	}

}
