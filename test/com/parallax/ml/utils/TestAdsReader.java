package com.parallax.ml.utils;

import static org.junit.Assert.*;

import org.junit.Test;

import com.parallax.ml.instance.BinaryClassificationInstance;
import com.parallax.ml.instance.BinaryClassificationInstances;

public class TestAdsReader {

	@Test
	public void testIrisReader() {
		BinaryClassificationInstances insts = AdsReader.readAds();
		for (BinaryClassificationInstance inst : insts) {
			assertEquals(inst.size(), 1558);
			assertTrue(inst.getLabel() != null);
			for (int x_i : inst) {
				assertTrue(inst.getFeatureValue(x_i) > 0);
			}
		}
	}


}
