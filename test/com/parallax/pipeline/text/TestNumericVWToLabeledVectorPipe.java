package com.parallax.pipeline.text;

import static org.junit.Assert.*;

import org.junit.Test;

import com.parallax.ml.vector.LinearVector;
import com.parallax.pipeline.Context;

public class TestNumericVWToLabeledVectorPipe {

	@Test
	public void testParses() {
		String example = "-1 | 13:3.96 24:3.47e-02 69:4.62 85:6.18e-02";

		Context<String> exampleContext = new Context<String>("", "", example);
		NumericVWToLabeledVectorPipe pipe = new NumericVWToLabeledVectorPipe(100);

		Context<LinearVector> exampleOut = pipe.operate(exampleContext);
		System.out.println(exampleOut.getLabel());
		System.out.println(exampleOut.getData());
		assertTrue(exampleOut.getLabel().equals("-1"));
		assertEquals(3.96, exampleOut.getData().getValue(13), 0.0000);
		assertEquals(3.47e-02, exampleOut.getData().getValue(24), 0.0000);
		assertEquals(4.62, exampleOut.getData().getValue(69), 0.0000);
		assertEquals(6.18e-02, exampleOut.getData().getValue(85), 0.0000);

	}

}
