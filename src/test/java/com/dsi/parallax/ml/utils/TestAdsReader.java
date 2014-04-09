package com.dsi.parallax.ml.utils;

import static org.junit.Assert.*;

import org.junit.Test;

import com.dsi.parallax.ml.instance.BinaryClassificationInstance;
import com.dsi.parallax.ml.instance.BinaryClassificationInstances;
import com.dsi.parallax.ml.utils.AdsReader;

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
