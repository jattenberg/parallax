package com.dsi.parallax.pipeline.csv;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import org.junit.Test;

import com.dsi.parallax.ml.util.csv.CSV;
import com.dsi.parallax.ml.vector.LinearVector;
import com.dsi.parallax.pipeline.Context;
import com.dsi.parallax.pipeline.csv.CSVPipe;

public class TestCSVPipe {

	File file = new File("./data/iris.data");

	@Test
	public void testReadsCsv() throws IOException {
		BufferedReader reader = new BufferedReader(new FileReader(file));
		reader.mark(10000);
		String[] parts = reader.readLine().split(",");
		int size = parts.length;
		reader.reset();
		
		CSV csv = CSV.numericColumnCSV(size).setLabelColumn(size - 1)
				.initialize();
		CSVPipe pipe = new CSVPipe(CSV.numericColumnCSV(size).setLabelColumn(size - 1)
				.initialize());
		String line;
		
		while (null != (line = reader.readLine())) {
			List<String> input = Arrays.asList(line.split(","));
			Context<LinearVector> works = csv.parseRow(input);
			Context<LinearVector> test = pipe.operate(Context.createContext(input));
			
			testCSVs(works, test);
			
		}
		
		

	}

	protected void testCSVs(Context<LinearVector> works,
			Context<LinearVector> test) {
		assertNull(works.getId());
		assertNull(test.getId());
		
		assertNotNull(works.getLabel());
		assertNotNull(test.getLabel());
		assertTrue(works.getLabel().equals(test.getLabel()));
		
		assertEquals(works.getData().size(), test.getData().size());
		for(int i = 0; i < works.getData().size(); i++) {
			assertEquals(works.getData().getValue(i), test.getData().getValue(i), 0);
		}
	}
}
