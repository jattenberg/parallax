package com.dsi.parallax.pipeline.csv;

import com.dsi.parallax.ml.util.csv.CSV;
import com.dsi.parallax.ml.vector.LinearVector;
import com.dsi.parallax.pipeline.Context;
import com.dsi.parallax.pipeline.FileSource;
import com.dsi.parallax.pipeline.Pipeline;
import com.dsi.parallax.pipeline.Source;
import com.dsi.parallax.pipeline.file.FileToLinesPipe;
import com.dsi.parallax.pipeline.text.StringSplitPipe;
import com.google.gson.reflect.TypeToken;

import java.io.File;
import java.lang.reflect.Type;
import java.util.Iterator;

public class CSVSource implements Source<LinearVector> {

	private static final long serialVersionUID = 2560396989298236707L;
	private final Pipeline<File, LinearVector> pipeline;

	public CSVSource(File file, CSV format, char delim) {
		pipeline = Pipeline.newPipeline(new FileSource(file))
				.addPipe(new FileToLinesPipe())
				.addPipe(new StringSplitPipe(delim))
				.addPipe(new CSVPipe(format));
	}

	@Override
	public Iterator<Context<LinearVector>> provideData() {
		return pipeline.process();
	}

	@Override
	public Type getType() {
		return new TypeToken<CSVSource>() {
		}.getType();
	}

}
