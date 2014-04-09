package com.dsi.parallax.pipeline.text;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import com.dsi.parallax.ml.vector.LinearVector;
import com.dsi.parallax.pipeline.Context;
import com.dsi.parallax.pipeline.text.SVMLightToLabeledVectorPipe;

public class TestSVMLightToLabeledVectorPipe {

	@Test
	public void testParses() {
		String example = "-1 1:0.43 3:0.12 9284:0.2";
		String exampleComment = "-1 1:0.43 3:0.12 9284:0.2 # abcdef";

		Context<String> exampleContext = new Context<String>("", "", example);
		SVMLightToLabeledVectorPipe pipe = new SVMLightToLabeledVectorPipe(9285);

		Context<LinearVector> exampleOut = pipe.operate(exampleContext);
		System.out.println(exampleOut.getData());
		assertTrue(exampleOut.getLabel().equals("-1"));
		assertEquals(0.43, exampleOut.getData().getValue(1), 0.0000);
		assertEquals(0., exampleOut.getData().getValue(2), 0.0000);
		assertEquals(0.12, exampleOut.getData().getValue(3), 0.0000);
		assertEquals(0.2, exampleOut.getData().getValue(9284), 0.0000);

		exampleContext = new Context<String>("", "", exampleComment);
		exampleOut = pipe.operate(exampleContext);
		assertTrue(exampleOut.getLabel().equals("-1"));
		assertEquals(0.43, exampleOut.getData().getValue(1), 0.0000);
		assertEquals(0., exampleOut.getData().getValue(2), 0.0000);
		assertEquals(0.12, exampleOut.getData().getValue(3), 0.0000);
		assertEquals(0.2, exampleOut.getData().getValue(9284), 0.0000);
	}

}
